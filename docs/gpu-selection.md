# catnip stage1 — GPU selection for spot training (Vast.ai / RunPod)

Workload: **yolo26n @ 640px, batch 32, 150 epochs** (~132k train images).
A tiny conv model — what matters is FPS-per-dollar, not VRAM or HBM bandwidth.
Rates = Vast.ai interruptible (from / median), captured Oct 2026.

| GPU | VRAM | Rate | Speed vs 4090 | Est. run cost | Tradeoff |
|---|---|---|---|---|---|
| **RTX 3090** | 24 GB | ~$0.08-0.15/h | ~0.6x | **$3-5** | Best $/run. Ampere (sm_86), rock-solid on cu126. Runs longer (~25-35h), more preemptions — irrelevant with the 10-min checkpoint loop. |
| **RTX 4090** | 24 GB | $0.15-0.60/h | 1.0x | **$3-10** | Default pick. Fastest consumer card for this. Most listings, easiest filters. |
| RTX 4080S | 16 GB | ~$0.2-0.35/h | ~0.75x | $4-7 | Fine but fewer listings; batch 32 fits, no headroom above ~48. |
| RTX A6000 | 48 GB | ~$0.25-0.45/h | ~0.5x | $8-13 | Big VRAM wasted on a nano model. Paying for memory you won't use. |
| A100 80GB | 80 GB | $0.40-0.84/h | ~0.8-1.0x | $6-21 | Datacenter-grade reliability, but the tiny model can't feed the HBM — money leaks. TF32 solid. |
| L40S | 48 GB | ~$0.6/h | ~1.1x | $10-13 | Reliable, but again buying VRAM you don't need. |
| H100 | 80 GB | $1.73+/h | ~1.4x | $27+ | Busts the $25 budget. Pointless for 2.6M params. |
| RTX 5090 | 32 GB | $0.45-0.86/h | ~1.3x | — | **Hard no for us**: Blackwell is sm_120, needs CUDA 12.8+; the pixi env is conda pytorch cu126 — it won't run. |
| V100 | 16 GB | ~$0.01-0.05/h | ~0.3x | $1-3 | Ancient but dirt cheap; fp16-only path. Only if feeling stingy. |

## Rules of thumb for this job

- **VRAM is not the lever** — yolo26n batch 32 uses ~8-10 GB. Extra VRAM only buys a bigger batch (diminishing returns past 64). Don't pay for it.
- **Datacenter cards underutilize** — their advantage is memory bandwidth and multi-tenant reliability, neither of which a nano YOLO consumes.
- **Speed per dollar: 3090 ≈ 4090 > everything else.** The 3090 costs less but runs 1.6x longer; on a $25 cap both fit easily.
- **Two-GPU instances**: ultralytics DDP works, but on this small model the sync overhead eats most of the gain — not worth paying for a 2nd GPU.

## Verdict

1. **RTX 4090** (reliability ≥95%, ≥32 GB RAM, ≥50 GB disk).
2. **RTX 3090** if 4090 listings look flaky or maximum cost certainty is wanted.
3. Bump `SPOT_BATCH=48` on either 24 GB card — ~15-25% throughput, free.

Prices float; check live listings at rent time (vast.ai/pricing).
