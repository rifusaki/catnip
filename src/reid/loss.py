"""Contrastive objectives for semi-supervised character clustering (plan §5.2).

``SupConLoss`` (Khosla et al., NeurIPS 2020) preserves per-character cluster
structure — each anchor contrasts against every in-batch sample, so recurring
non-Izutsumi characters can be pulled into distinct pseudo-clusters instead of
one heterogeneous blob.  ``SubCenterArcFace`` (InsightFace) adds an angular
margin capacity of K sub-centers per class so Izutsumi's visual modes (chibi,
hooded, silhouette) can occupy separate sub-centers without penalty.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Numerical-stability floor for sqrt/arccos arguments.
_EPS = 1e-7


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss over L2-normalized features.

    loss = mean over anchors having >= 1 same-class in-batch positive of
    (1/|P(i)|) * sum_{p in P(i)} -log( exp(z_i . z_p / tau) / sum_{a != i}
    exp(z_i . z_a / tau) ).  Anchors with no in-batch positive contribute
    exactly 0 (they are dropped from the mean), so batches of all-unique
    labels yield an exact 0 loss instead of NaN.  With ``labels=None`` the
    loss falls back to instance discrimination (every sample is its own
    class).

    When every anchor shares one label, the in-batch denominator still counts
    same-class competitors: an identical same-class batch of size B converges
    to log(B - 1), not 0.  Only a matched pair (B = 2) is exactly 0.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
        batch_size = features.shape[0]
        if batch_size < 2:
            # No other sample exists to contrast against; keep the graph.
            return (features * 0.0).sum()

        # Law 2: features are promised L2-normalized; re-normalizing is an
        # idempotent parse making the sim range [-1, 1] unconditionally true.
        features = F.normalize(features, dim=1)
        if labels is None:
            labels = torch.arange(batch_size, device=features.device)
        if labels.shape[0] != batch_size:
            raise ValueError(
                f"labels length {labels.shape[0]} != batch size {batch_size}"
            )

        anchor_to_sample = features @ features.T / self.temperature
        # Row-wise max subtraction keeps exp() in range for small tau.
        anchor_to_sample = anchor_to_sample - anchor_to_sample.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = same_class & ~eye          # P(i)
        exclude_self = ~eye                        # denominator: all a != i

        exp_scores = torch.exp(anchor_to_sample) * exclude_self
        log_denominator = torch.log(exp_scores.sum(dim=1) + _EPS)
        log_prob = anchor_to_sample - log_denominator.unsqueeze(1)

        has_positive = positive_mask.any(dim=1)
        if not has_positive.any():
            return (features * 0.0).sum()

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        per_anchor_loss = torch.where(has_positive, -mean_log_prob_pos, torch.zeros_like(mean_log_prob_pos))
        return per_anchor_loss[has_positive].mean()


class SubCenterArcFace(nn.Module):
    """Sub-Center ArcFace: K sub-centers per class with an additive angular margin.

    forward(features, labels) returns (logits, loss):

    * weight rows: [out_features * K, in_features], L2-normalized.
    * cosine: for every class, the sub-center selection is winner-take-all —
      the K subcenter with the highest cosine to the feature represents the
      class, so multi-modal classes (Izutsumi's chibi/hooded/silhouette
      modes) can specialize their own sub-centers.
    * margin: among the true class's K sub-centers, the argmax sub-center is
      selected and rotated by the additive angular margin m via the standard
      cos(theta + m) formulation with the hard-sample switch
      (phi = cos(theta)cos(m) - sqrt(1 - cos(theta)^2) sin(m) while
      cos(theta) > cos(pi - m), else phi = cos(theta) - sin(pi - m) * m).
    * logits = s * cos(theta (+ m)) and loss = cross-entropy(logits, labels).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_subcenters: int = 5,
        s: float = 64.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        if num_subcenters < 1:
            raise ValueError(f"num_subcenters must be >= 1, got {num_subcenters}")
        if s <= 0.0:
            raise ValueError(f"scale s must be > 0, got {s}")
        if not 0.0 <= m < math.pi:
            raise ValueError(f"margin m must lie in [0, pi), got {m}")

        self.in_features = in_features
        self.out_features = out_features
        self.num_subcenters = num_subcenters
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.cos_pi_minus_m = math.cos(math.pi - m)
        self.sin_pi_minus_m = math.sin(math.pi - m)
        # f(m): the hard-sample substitute keeps phi < cos(theta) for angles
        # where cos(theta + m) would cross behind -1 (theta > pi - m).
        self.hard_sample_offset = self.sin_pi_minus_m * m

        self.weight = nn.Parameter(torch.empty(out_features * num_subcenters, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() != 2 or features.shape[1] != self.in_features:
            raise ValueError(
                f"features must be [B, {self.in_features}], got {tuple(features.shape)}"
            )
        batch_size = features.shape[0]
        if labels.shape[0] != batch_size:
            raise ValueError(
                f"labels length {labels.shape[0]} != batch size {batch_size}"
            )
        if (labels < 0).any() or (labels >= self.out_features).any():
            raise ValueError(
                f"labels must lie in [0, {self.out_features}), got {labels.tolist()}"
            )

        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)

        cosine = F.linear(features, weight).view(
            batch_size, self.out_features, self.num_subcenters
        )  # [B, C, K]
        # Winner-take-all sub-center per class.
        base_cosine = cosine.max(dim=2).values  # [B, C]

        row_indices = torch.arange(batch_size, device=features.device)
        true_class_cosines = cosine[row_indices, labels]  # [B, K]
        winning_subcenter = true_class_cosines.argmax(dim=1)  # [B]
        true_cosine = true_class_cosines[row_indices, winning_subcenter]  # [B]

        # cos(theta + m) via the clamp/sqrt formulation. arccos first for the
        # hard-sample switch, sqrt(1 - cos^2) for the sine term.
        cos_theta = true_cosine.clamp(-1.0 + _EPS, 1.0 - _EPS)
        sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=0.0))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        phi = torch.where(
            cos_theta > self.cos_pi_minus_m,
            phi,
            cos_theta - self.hard_sample_offset,
        )

        logits = base_cosine.clone()
        logits[row_indices, labels] = phi
        logits = self.s * logits
        return logits, F.cross_entropy(logits, labels)
