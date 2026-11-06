"""Character embedding backbone with Generalized Mean (GeM) pooling and multi-granularity fusion.

Implements plan task 5.1 (docs/implementation_plan.md §5): a pluggable
anime/manga-pretrained feature backbone combined with learnable GeM pooling
and a linear projection to a 256-d L2-normalized embedding, plus the
confidence-weighted fusion of per-part (face/head/body) embeddings into a
single character-instance embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical input resolution for ``embed_single`` and Stage 2 training
# preprocessing (matches settings.params.reid.img_size and the ConvNeXt
# ImageNet pretraining resolution).
REID_IMG_SIZE = 224

# GeM clamps activations to this floor before exponentiation so gradients of
# pow() stay finite for negative ConvNeXt/DINOv2 activations (GeM paper §3).
_GEM_EPS = 1e-6

# ImageNet color statistics used by both the torchvision backbones and
# ``embed_single`` preprocessing (must match: backbone pretraining assumed
# these; swapping them subtly corrupts embeddings).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Fusion treats vectors shorter than this as the zero vector (a fusion of
# exclusively zero-confidence/zero-weight parts is unembeddable).
_FUSION_MIN_NORM = 1e-12


def preprocess_rgb(image_rgb: np.ndarray, img_size: int) -> torch.Tensor:
    """Convert a uint8 RGB image into a normalized [3, img_size, img_size] tensor.

    Resize (bilinear, antialiased), scale to [0, 1], and standardize with the
    ImageNet mean/std the pretrained backbones were fit against.  This is the
    single preprocessing path shared by ``CharacterEmbedder.embed_single`` and
    the Stage 2 training loop, so embeddings never drift apart.
    """
    pixels = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1)))
    pixels = pixels.unsqueeze(0).to(torch.float32) / 255.0

    if pixels.shape[-2:] != (img_size, img_size):
        pixels = F.interpolate(
            pixels,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    mean = _IMAGENET_MEAN
    std = _IMAGENET_STD
    if pixels.device != mean.device:
        mean, std = _IMAGENET_MEAN.to(pixels.device), _IMAGENET_STD.to(pixels.device)
    return ((pixels - mean) / std).squeeze(0)


def _build_convnext_small_features(pretrained: bool) -> nn.Module:
    """Torchvision ConvNeXt-Small truncated to its convolutional stem+blocks.

    ``features`` output on 224x224 input is [B, 768, 7, 7] (verified against
    torchvision 0.21); the avgpool/classifier head is discarded because GeM
    replaces average pooling.
    """
    from torchvision.models import ConvNeXt_Small_Weights, convnext_small

    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    return convnext_small(weights=weights).features


def _build_dinov2_patch_tokens(pretrained: bool) -> nn.Module:
    """DINOv2 ViT-S/14 patch-token extractor loaded lazily via torch.hub.

    Emits [B, C, N, 1] (patch tokens reindexed as a "spatial" axis) so GeM
    pools one feature map regardless of backbone family.  Weight download
    requires network access — this backbone is intended for Colab training
    runs, never for offline tests.
    """
    del pretrained  # torch.hub always fetches the pretrained checkpoint.
    try:
        from torch.hub import load as torch_hub_load
    except Exception as exc:  # pragma: no cover - torch.hub always imports
        raise RuntimeError(
            "backbone 'dinov2_vits14' requires torch.hub, which is unavailable: "
            f"{exc}. Use the default 'convnext_small' backbone offline."
        ) from exc

    try:
        backbone = torch_hub_load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception as exc:
        raise RuntimeError(
            "backbone 'dinov2_vits14' could not be loaded via "
            "torch.hub('facebookresearch/dinov2', 'dinov2_vits14') — this "
            "requires network access (run on Colab) or a populated torch hub "
            f"cache. Original error: {exc}"
        ) from exc

    return _DinoV2PatchTokens(backbone)


class _DinoV2PatchTokens(nn.Module):
    """Adapt a DINOv2 ViT into a [B, C, N, 1] patch-token feature map.

    The CLS token is a class summary, not a spatial feature — the plan's GeM
    pooling operates over patch tokens only.
    """

    def __init__(self, dinov2: nn.Module) -> None:
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.dinov2, "forward_features"):
            raise RuntimeError(
                "loaded DINOv2 model does not expose forward_features(); "
                "the patch-token contract of this adapter is unmet"
            )
        features = self.dinov2.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # [B, N, C]
        return patch_tokens.transpose(1, 2).unsqueeze(-1)


def _build_ccip(pretrained: bool) -> nn.Module:
    """deepghs/ccip reserved for later integration — refuse to fake it."""
    del pretrained
    raise NotImplementedError(
        "backbone 'ccip' (deepghs/ccip) is not integrated yet; use "
        "'convnext_small' (default) or 'dinov2_vits14' for now. Tracked under "
        "AGENTS.md §7.2 / plan task 5.1."
    )


# Backbone registry: name -> feature-map builder (all builders emit
# [B, C, H, W] feature maps ready for GeM).
BACKBONES: dict[str, Callable[[bool], nn.Module]] = {
    "convnext_small": _build_convnext_small_features,
    "dinov2_vits14": _build_dinov2_patch_tokens,
    "ccip": _build_ccip,
}

# Feature channels per backbone (backbone_dim of the projection head).
BACKBONE_DIMS = {
    "convnext_small": 768,  # ConvNeXt-S final stage width (verified on tv 0.21)
    "dinov2_vits14": 384,  # ViT-S/14 embed dim
}


class GeneralizedMeanPooling(nn.Module):
    """Learnable Generalized Mean pooling (GeM) over spatial dims (2, 3).

    pooled = (mean(clamp(x, min=eps)^p))^(1/p) with p learnable (init from
    ``gem_p``, typically 3.0).  p interpolates between mean (p=1) and max
    (p->inf) pooling, sharpening focus on the most salient regions of a
    character crop.

    p is clamped to >= 1.0 in ``forward``: p < 1 over-amplifies near-floor
    activations and inflates gradient variance of pow()/pow(1/p), destabilizing
    optimization.  The clamp keeps p learnable while pinning the unstable
    region away; its gradient is exactly zero while clamped.
    """

    def __init__(self, p: float = 3.0) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exponent = self.p.clamp(min=1.0)
        clamped = x.clamp(min=_GEM_EPS)
        return clamped.pow(exponent).mean(dim=(2, 3)).pow(1.0 / exponent)


class CharacterEmbedder(nn.Module):
    """L2-normalized character embedding: backbone -> GeM -> linear head.

    forward([B, 3, H, W]) -> [B, embedding_dim] with unit rows.  ``embed_single``
    ingests an everyday (H, W, 3) uint8 RGB array and returns a
    ``(embedding_dim,)`` float32 unit vector, so callers never hand-write
    preprocessing.
    """

    def __init__(
        self,
        backbone: str = "convnext_small",
        embedding_dim: int = 256,
        gem_p: float = 3.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # --- Parse at boundary: registry-controlled backbone names only
        if backbone not in BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Supported: {sorted(BACKBONES)}"
            )
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if gem_p < 1.0:
            raise ValueError(f"gem_p must be >= 1.0 for stable GeM, got {gem_p}")

        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.img_size = REID_IMG_SIZE
        # Weights download only when pretrained=True (tests always pass False).
        # ccip raises NotImplementedError here; dims are resolved only after a
        # builder succeeds, since unimplemented backbones have no known width.
        self.backbone = BACKBONES[backbone](pretrained)
        self.backbone_dim = BACKBONE_DIMS[backbone]
        self.pool = GeneralizedMeanPooling(gem_p)
        self.projection_head = nn.Linear(self.backbone_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)
        if feature_map.dim() != 4:
            raise ValueError(
                f"backbone '{self.backbone_name}' must emit [B, C, H, W], got "
                f"shape {tuple(feature_map.shape)}"
            )
        pooled = self.pool(feature_map)
        return F.normalize(self.projection_head(pooled), dim=1)

    @torch.no_grad()
    def embed_single(self, image_rgb: np.ndarray) -> np.ndarray:
        """Embed one (H, W, 3) uint8 RGB image into a unit-length float32 vector."""
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(
                f"image_rgb must be a numpy array, got {type(image_rgb).__name__}"
            )
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                "image_rgb must have shape (H, W, 3), got "
                f"{tuple(image_rgb.shape)}"
            )
        if image_rgb.dtype != np.uint8:
            raise ValueError(
                f"image_rgb must be uint8 (0-255 range), got dtype "
                f"{image_rgb.dtype}"
            )

        device = self.projection_head.weight.device
        batch = preprocess_rgb(image_rgb, self.img_size).unsqueeze(0).to(device)
        embedded = self.forward(batch)
        return embedded.squeeze(0).detach().cpu().numpy().astype(np.float32)


def fuse_part_embeddings(
    parts: dict[str, np.ndarray],
    scores: dict[str, float],
    weights: dict[str, float],
) -> np.ndarray:
    """Fuse present-part embeddings into one unit-length instance embedding.

    Implements plan §5.1: e = L2Norm(sum_c w'_c * s_c * e_c) where w' are the
    configured part weights renormalized over the PRESENT parts only, so
    head-only and body-only crops still embed validly in the shared space.

    Args:
        parts: present parts only — {"face"|"head"|"body": unit-length vector}.
        scores: per-part detector confidence; keys must match ``parts`` exactly.
        weights: full configured part fusion weight map (superset allowed).

    Raise:
        ValueError: for empty/missing/mismatched inputs or an all-zero fusion.
    """
    if not parts:
        raise ValueError("fuse_part_embeddings requires at least one part")
    if set(parts) != set(scores):
        raise ValueError(
            f"parts and scores keys must match exactly; parts={sorted(parts)}, "
            f"scores={sorted(scores)}"
        )

    missing_weights = set(parts) - set(weights)
    if missing_weights:
        raise ValueError(
            f"No fusion weight configured for present parts: "
            f"{sorted(missing_weights)}; weights={sorted(weights)}"
        )

    part_names = sorted(parts)
    matrices = np.stack([np.asarray(parts[name], dtype=np.float32) for name in part_names])
    if matrices.ndim != 2:
        raise ValueError(
            "part embeddings must be 1-D vectors of equal length, got lengths "
            f"{[len(np.asarray(parts[name]).ravel()) for name in part_names]}"
        )
    scores_vec = np.array([float(scores[name]) for name in part_names], dtype=np.float32)
    weight_vec = np.array([float(weights[name]) for name in part_names], dtype=np.float32)

    # Renormalize the configured weights over the present parts so the fused
    # vector stays in a predictable scale relative to full instances.
    present_weight_sum = weight_vec.sum()
    if present_weight_sum <= 0.0:
        raise ValueError(f"Fusion weights for present parts sum to <= 0: {dict(zip(part_names, weight_vec.tolist()))}")

    renormalized_weights = weight_vec / present_weight_sum
    fused = ((renormalized_weights * scores_vec)[:, None] * matrices).sum(axis=0)

    fused_norm = float(np.linalg.norm(fused))
    if fused_norm < _FUSION_MIN_NORM:
        raise ValueError(
            "Fused embedding has zero magnitude (every present part carries "
            f"zero weight or zero confidence); parts={dict(zip(part_names, scores_vec.tolist()))}"
        )
    return (fused / fused_norm).astype(np.float32)


def save_embedder(model: CharacterEmbedder, model_path: str | Path) -> None:
    """Persist construction config + CPU state_dict for ``load_embedder``.

    Parameters:
        model: the trained ``CharacterEmbedder`` (any device).
        model_path: destination .pt path; parent directories are created.
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "backbone": model.backbone_name,
        "embedding_dim": model.embedding_dim,
        "gem_p": float(model.pool.p.item()),
        "img_size": model.img_size,
        "backbone_dim": model.backbone_dim,
    }
    state_dict = {
        key: value.detach().to("cpu") for key, value in model.state_dict().items()
    }
    torch.save({"config": config, "state_dict": state_dict}, path)


def load_embedder(model_path: str | Path) -> CharacterEmbedder:
    """Rebuild a saved ``CharacterEmbedder`` (eval mode, CPU, no network).

    The backbone is always rebuilt with ``pretrained=False``: the saved
    state_dict carries all weights, so loading never re-downloads.

    Raise:
        FileNotFoundError: checkpoint does not exist.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Embedder checkpoint not found: {path} — train one with "
            "`pixi run train-stage2` first"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = {"backbone", "embedding_dim", "gem_p", "img_size", "backbone_dim"} - set(
        checkpoint.get("config", {})
    )
    if missing:
        raise ValueError(f"Checkpoint config is malformed; missing keys {sorted(missing)}")

    config = checkpoint["config"]
    model = CharacterEmbedder(
        backbone=config["backbone"],
        embedding_dim=config["embedding_dim"],
        gem_p=config["gem_p"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
