# Plan — Migrate catnip storage from GCS to Cloudflare R2 + onboard Lightning AI Studio

## Status: DATA-PARITY COMPLETE (2026-10-14) — R2 == GCS 1:1; Phase 4 (Lightning) + Phase 5 (decommission) pending

## Implementation progress

- ✅ **Phase 1 (HDD audit):** HDD online; `training/stage1_sliced` confirmed complete & identical to GCS (train 132,435 / val 16,264 / test 15,993, all images==labels). HDD is a partial working copy (only `data/staging/`, `training/*`, `models/`); source dirs + `runs/` live in GCS. HDD+`.DS_Store` removed. Found `stage1_sliced_safe` (same counts) and `stage1_sliced_safe_v2` (129,933/15,949/15,696) NSFW-filtered variants on HDD. Audit report: `.opencode/plan/state/2026-09-03-audit.json`.
- ✅ **Phase 2 (wrangler + bucket):** wrangler v4.128.0 authenticated (account `rifusaki`); bucket `catnip-canonical` created; `object put/get/delete` round-trip verified via OAuth.
- ✅ **Phase 3 cleanup:** GCS `.DS_Store`×24 deleted; stale `runs/detect/stage1-{2..9}` deleted (kept `stage1-10`, `stage1`); stale `compressed/stage1_sliced.tar` deleted. GCS service account `catnip-sippy` (objectViewer) + JSON key at `~/secrets/catnip-gcs-svc.json` created.
- ✅ **Phase 3 (Sippy):** R2 API token supplied (`~/secrets/r2-sippy.conf`); `wrangler r2 bucket sippy enable catnip-canonical --provider GCS --bucket catnip-data` succeeded 2026-09-03; **read-through lazy-copy verified** (dataset.yaml fetched on demand, GCS metadata preserved). ⚠️ **Correction (2026-10-13):** Sippy has **no background backfill** — it copies objects only when requested (per Cloudflare docs). The earlier "server-side backfill running" assumption was wrong.
- 🔄 **Fresh payload tarball:** ✅ **IN R2** `s3://catnip-canonical/compressed/stage1_sliced.tar` (multipart, 14,018,147,328 B — size-verified against local). ⚠️ wrangler `object put` refused it ("upload too large", single-part limit) → uploaded via **`s5cmd` S3 multipart** with the R2 token (must map `R2_ACCESS_KEY_ID/SECRET` → `AWS_ACCESS_KEY_ID/SECRET`; env-named `AWS_*` required, otherwise s5cmd falls back to EC2 IMDS and hangs).
- ⚠️ **2026-10-13 status:** R2 holds only the 3 objects ever requested (`compressed/stage1_sliced.tar` [manual upload], `training/stage1_sliced/dataset.yaml`, `models/yolo26n.pt`). Sippy is enabled and functional but will never bulk-copy. **Decision: bulk migration via Super Slurper** (dashboard-driven, GCS JSON key = `~/secrets/catnip-gcs-svc.json`), 3 concurrent job limit → split by prefix. Sippy stays enabled as straggler safety net until decommission.
- **GCS baseline for verification (measured 2026-10-13, post-cleanup, live listing):** `data/` 325,364 · `training/` 496,377 (stage1_sliced 329,385 + stage1 166,992) · `models/` 3 · `runs/` 8 · **total 821,752** (compressed/ and root droppings gone). Expect R2 = 821,752 + the pre-existing tarball = 821,753 after Slurper completes with "Skip existing".
- ✅ **2026-10-14 migration complete & verified:** Super Slurper jobs A (`data/`), B (`training/`), C (whole-bucket, Skip-existing) all finished. Slurper skips zero-byte objects — 8,467 empty label files (`data/` 3,142; `stage1_sliced` labels 5,325) were backfilled via direct `s5cmd run` upload (quote s3 URLs in the run file — keys contain apostrophes). **Verification: full GCS↔R2 name-listing diff = 0 missing, 1 expected extra (`compressed/stage1_sliced.tar`); R2 holds 821,747 objects (821,746 GCS real objects + tarball; the 6 zero-byte GCS "objects" beyond that are directory markers, not files); sliced splits 1:1 (train 132,435 / val 16,264 / test 15,993, images == labels).** Sippy is still enabled — **disable before GCS deletion** (§3.5/§5.3 ordering).
- 🔄 **Phase 4 (Lightning Studio):** Studio ingest done 2026-10-14 — `compressed/stage1_sliced.tar` + `models/yolo26n.pt` pulled from R2 via wrangler and extracted. Gotcha hit: wrangler `object get/put` **requires `--remote`**, otherwise it reads local dev storage and fails with "key does not exist". Lightning storage lesson: `/teamspace` root is not user-writable and `uploads/` is read-only from Studio; working copy goes in **Studio home `~/catnip-data`** (persisted across restarts; R2 stays canonical). S3/R2 connector mounts (`/teamspace/{s3,r2}_connections/`) are read-only FUSE — browsing only, never train through them. Next: env setup + train.
- 🔄 **Phase 5 (decommission) started 2026-10-14:** **Sippy disabled** (`wrangler r2 bucket sippy disable`) — parity verified first, so the lazy-copy safety net is no longer needed. §5.1 code/config cleanup **done**: `reorganize_gcs.py` deleted (+ pixi task), `kaggle_sync.py` stripped of gs:///rclone remote paths (local-source only, `--archive-uri`/`--source-uri`/`--skip-gcs` removed), Colab notebook rewritten to R2/boto3, runbook + architecture + `.env.example` + `pipeline.yaml` de-GCS'd, `gcp-mcp` dropped from apm.yml. Repo grep = 0 functional GCS refs (only historical notes here). **Remaining gates before bucket deletion:** §5.2 Label Studio URL inventory/re-point, ≥1 epoch trained from R2 data, ~7-day grace.

Decisions captured from the user this session:

- **Compute:** Lightning AI Studio is the primary training host going forward (Kaggle demoted to optional).
- **Storage:** Cloudflare R2 replaces GCS entirely as the canonical archive + compute source. GCS egress costs were the trigger; R2 has **$0 egress**.
- **GCS:** will be decommissioned (user explicitly does not care about keeping it).
- **HDD (`/Volumes/rifuSSD/catnip-data`):** assumed plugged in. **Audit HDD vs GCS first** before any copy.
- **Tooling: rclone is dropped.** All R2 operations use **`wrangler`** (`npx wrangler`, already authenticated, v4.128.0). The GCS→R2 bulk migration runs **server-side via Sippy** (`wrangler r2 bucket sippy`, GCS provider) — no 59 GB local upload.

---

## Current state (measured 2026-09-03)

### GCS `gs://catnip-data` — 59.24 GB, 821,801 objects (verified live, full listing)

| Prefix | Objects | Size | Notes |
|---|---|---|---|
| `compressed/stage1_sliced.tar` | 1 | 14.0 GB | Legacy Colab tarball (useful single-file artifact — verify freshness, see Phase 1) |
| `training/stage1_sliced/` | 329,385 | 13.7 GB | ✅ **complete 1:1** (train 132,435/132,435; val 16,264/16,264; test 15,993/15,993). 5,325 zero-byte = empty label files (negative slices) |
| `training/stage1/` (unsliced unified) | 166,992 | 9.9 GB | ✅ complete 1:1 (66,796 / 8,350 / 8,351) |
| `data/staging/` | 166,993 | 9.8 GB | 🔄 regenerable (`scripts/unify/*.py`) |
| `data/anime_head_detection/` | 136,688 | 5.4 GB | source v1/v2/ani_face |
| `data/manga109/` | 10,715 | 4.2 GB | 10,713 imgs + 110 XML |
| `data/izutsumi/` | 2,829 | 1.6 GB | 2,768 manga imgs + 77 annotation JSONs |
| `data/AnimeHeadsv3/` | 8,161 | 0.6 GB | COCO source |
| `models/` | 3 | 0.07 GB | pretrained `yolo26{n,m,s}.pt` only — no trained deliverable yet |
| `runs/detect/` | 31 | ~0 | stale `stage1-2..10` runs (runbook: keep last only) |
| `.DS_Store` + root files | ~12 | ~0 | macOS droppings throughout `data/**` |

State captured this session:
1. **`training/stage1_sliced` on GCS is complete and train-ready** (images == labels 1:1 on all splits; confirmed from the full recursive listing). An earlier mid-stream reading suggested broken labels — that was an artifact of a partially-written listing, now corrected.
2. **Kaggle has no usable dataset version** (v1–v5 zombies, v6/v7 NSFW-auto-unpublished). Moot if Lightning becomes primary — the NSFW filter stays optional (Phase 6).
3. **`wrangler r2` ships bucket + single-object commands only** (no recursive list/sync/check). Bulk migration therefore uses **Sippy** (server-side GCS→R2); single-file transfers use `wrangler r2 object put/get`; object counts for verification come from the R2 dashboard (or an optional S3-API `s5cmd` check with a scoped R2 token).
4. **GCS hygiene:** `.DS_Store` droppings throughout `data/**`, 9 stale `runs/detect/stage1-*` runs, no trained model deliverable yet.

### Tooling note
- `npx wrangler` v4.128.0, already authenticated (`npx wrangler whoami` to confirm). Subcommands used:
  - `wrangler r2 bucket create|list|info`
  - `wrangler r2 bucket sippy enable --provider GCS --bucket <gcs> --service-account-key-file <json>` / `sippy get` / `sippy disable`
  - `wrangler r2 object put <bucket>/<key> --file <path> --remote`
  - `wrangler r2 object get <bucket>/<key> --file <path> --remote`
- ⚠️ **wrangler object ops default to LOCAL (dev) storage** — always pass `--remote`.
- ⚠️ **wrangler `object put` is single-part → refuses files >5 GiB** ("r2 object upload too large"). Use **`s5cmd`** (brew, v2.x) or `aws s3 cp` against the R2 S3 endpoint (multipart) for the 13 GB payload tarball:
  ```bash
  set -a; source ~/secrets/r2-sippy.conf; set +a; export AWS_REGION=auto
  s5cmd --endpoint-url https://<accountid>.r2.cloudflarestorage.com \
    cp stage1_sliced.tar s3://catnip-canonical/compressed/stage1_sliced.tar
  ```
  (`s5cmd`/`aws` are also the practical way to *count/list* objects, since wrangler has no `r2 object ls`.)
- The old rclone `r2:` remote and its credentials are **retired** — do not use or reference them (rotate/revoke the previously-exposed R2 keys for hygiene; see Phase 2).

---

## Target state

```
                     R2 (catnip-canonical)   ← canonical archive + compute source ($0 egress)
                     ├─ data/                ← sources + staging (mirrors cleaned GCS tree)
                     ├─ training/            ← stage1 + stage1_sliced (complete 1:1 labels)
                     ├─ models/              ← pretrained + trained deliverable
                     ├─ runs/detect/stage1/  ← latest run only
                     └─ compressed/stage1_sliced.tar   ← the artifact Lightning pulls
        ▲                                        ▲
HDD (dev origin, local working copy)      Lightning AI Studio (T4/L4 GPU)
                                           └─ wrangler r2 object get → tar -xf → train --resume
GCS ── Sippy (server-side) ──► R2 ── GCS deleted after grace period + sippy disable
```

**Key principle:** R2 keeps the **same prefix layout** as GCS so `config/pipeline.yaml` paths don't churn. R2 egress = $0 → Lightning (and any future host) can re-pull the 14 GB sliced payload every session for free.

---

## Phase 0 — Preflight decisions

- [ ] Bucket layout: single bucket **`catnip-canonical`** (private) for everything; create a second public `catnip-serve` only when the home server/Label Studio needs URL access (deferred).
- [ ] Authoritativeness: **GCS → R2 via Sippy** for the archive copy; **HDD remains the working origin** (any drift resolved by HDD title before Sippy runs); GCS retired after R2 is verified.
- [ ] Kaggle status: if Lightning becomes the only training host, do **not** spend effort on Kaggle/NSFW filtering now; park `nsfw_report.json` (Phase 6).
- [ ] Label Studio dependency: `label.rifusaki.com` imports images from **GCS URLs** (`architecture.md`). Inventory and re-point or mirror before GCS deletion (Phase 5).

---

## Phase 1 — HDD audit (HDD assumed plugged in)

Goal: confirm the local working copy matches the (complete) GCS dataset, decide which artifacts need rebuilding (esp. the tarball), and find GCS-only objects.

```bash
# 1. Mount + capacity
ls /Volumes/rifuSSD/ && df -h /Volumes/rifuSSD

# 2. Top-level inventory (sizes + object counts)
du -sh /Volumes/rifuSSD/catnip-data/* 2>/dev/null
find /Volumes/rifuSSD/catnip-data -type f | wc -l

# 3. GATE — sliced images:labels ratio, per split (must match GCS exactly)
for s in train val test; do
  im=$(find /Volumes/rifuSSD/catnip-data/training/stage1_sliced/images/$s -type f | wc -l)
  lb=$(find /Volumes/rifuSSD/catnip-data/training/stage1_sliced/labels/$s  -type f | wc -l)
  echo "split=$s images=$im labels=$lb"   # expect 132435 / 16264 / 15993 on both
done

# 4. macOS droppings sweep (so they never reach R2)
find /Volumes/rifuSSD/catnip-data -name ".DS_Store" | wc -l

# 5. Tarball gate — is compressed/stage1_sliced.tar current (dataset.yaml + 1:1 imgs/labels)?
tar -tf /Volumes/rifuSSD/catnip-data/compressed/stage1_sliced.tar 2>/dev/null | grep -c "images/" 
tar -tf /Volumes/rifuSSD/catnip-data/compressed/stage1_sliced.tar 2>/dev/null | grep -c "labels/"
#    If stale/missing → rebuild it from the local sliced tree:
#    tar -C /Volumes/rifuSSD/catnip-data -cf compressed/stage1_sliced.tar.$(date +%Y%m%d) training/stage1_sliced

# 6. Diff vs GCS (capture already saved at /tmp/catnip-gcs-full.txt)
#    → list HDD-only and GCS-only paths to decide what must survive into R2.
```

**Gates:**
- Split counts must equal GCS baseline (132,435 / 16,264 / 15,993). If the HDD differs → reconcile from GCS copy before Phase 3 (do **not** rebuild the dataset unless actually broken).
- `compressed/stage1_sliced.tar` (the Lightning payload) must contain `dataset.yaml`, 164,692 `images/`, and 164,692 `labels/` entries. Rebuild if not.

Output: write `audit-report.json` (per-dir sizes, per-split ratios, tarball verdict, GCS-only list).

---

## Phase 2 — R2 bucket + wrangler auth

1. **Confirm auth:** `npx wrangler whoami` (should show the Cloudflare account).
2. **Create the bucket:**
   ```bash
   npx wrangler r2 bucket create catnip-canonical
   npx wrangler r2 bucket list
   npx wrangler r2 bucket info catnip-canonical
   ```
3. **Credentials hygiene (segmented by where they're used):**
   - *This dev machine:* wrangler OAuth already in place — nothing to do.
   - *Lightning Studio (pull side):* needs Cloudflare auth. Create a **`CLOUDFLARE_API_TOKEN`** (Workers R2 Storage → Edit, scoped to a user with the bucket) and add it as a Lightning secret, or run `npx wrangler login` inside the Studio once.
   - *Optional (only if an S3-API verification tool is chosen):* create an **R2 API token** (`Object Read` on `catnip-canonical`) for `s5cmd`/`aws s3` `ls` counts. Not required for the wrangler-only path.
   - **Rotate/revoke the old rclone R2 keys** that were printed in session logs earlier — rclone is dropped, so also remove/comment the `[r2]` and `[gcs]` entries from `~/.config/rclone/rclone.conf` to prevent accidental use.
4. **No `rclone.conf` edits, no R2 S3 keys for the primary path.**

---

## Phase 3 — Migrate GCS → R2 via Sippy (server-side)

### 3.1 Clean GCS first (so junk never lands in R2)
Sippy copies everything in the source bucket, so pre-clean:
```bash
# .DS_Store droppings (paths captured in /tmp/catnip-gcs-full.txt)
gcloud storage ls -r gs://catnip-data 'gs://catnip-data/**/.DS_Store' | gcloud storage rm -I

# stale runs: delete runs/detect/stage1-{2..9} (keep the most recent per runbook)
gcloud storage rm -r gs://catnip-data/runs/detect/stage1-2
# ... repeat for the other stale run dirs (re-verify which is "latest" at audit time)
```

### 3.2 GCS service account for Sippy
1. In GCP: create a service account, grant **Storage Object Viewer** on the bucket (or project-scoped), download the JSON key to `~/secrets/catnip-gcs-svc.json` (ensure it's gitignored/not committed).
2. Verify the key can read: `gcloud auth activate-service-account --key-file=...` or just proceed to Sippy.

### 3.3 Enable Sippy
```bash
npx wrangler r2 bucket sippy enable catnip-canonical \
  --provider GCS \
  --bucket catnip-data \
  --service-account-key-file ~/secrets/catnip-gcs-svc.json
```
Sippy provides **on-demand lazy-copy only** — a GET of a not-yet-copied object is served from GCS and copied to R2. There is **no server-side bulk backfill** (corrected 2026-10-13); bulk copy is done by **Super Slurper** (dashboard: R2 → Data migration).

### 3.4 Monitor + verify
```bash
npx wrangler r2 bucket sippy get catnip-canonical    # backfill status / drain state
```
- **Count/bytes:** R2 dashboard "catnip-canonical" object count + storage should converge toward the cleaned GCS baseline (~821k − droppings; ~59 GB).
- **Spot-check objects:** 
  ```bash
  npx wrangler r2 object get catnip-canonical/training/stage1_sliced/dataset.yaml --file /tmp/ds.yaml
  npx wrangler r2 object get catnip-canonical/models/yolo26n.pt --file /tmp/yolo26n.pt
  npx wrangler r2 object get catnip-canonical/training/stage1_sliced/images/train/$(ls | head -1) ...
  ```
- **Dataset gate:** confirm `dataset.yaml` content matches GCS; for a real per-split 1:1 count use the R2 dashboard (or optional scoped-R2-token + `s5cmd ls --recursive` count — a one-off verification, still no rclone).
- **Proceed only when:** cleaned-GCS object count ≈ R2 object count, spot checks match, and the sliced split counts are 1:1 in R2 (train 132,435 / val 16,264 / test 15,993).

### 3.5 Freeze after verification
```bash
npx wrangler r2 bucket sippy disable catnip-canonical
```
(Do this **before** deleting GCS, per the Sippy lifecycle.)

> **Alternative (only if Sippy can't be used):** push the payload as single artifacts with wrangler:
> ```bash
> npx wrangler r2 object put catnip-canonical/compressed/stage1_sliced.tar --file /Volumes/rifuSSD/catnip-data/compressed/stage1_sliced.tar
> ```
> and mirror the non-sliced trees with an S3-API bulk tool (`s5cmd`, *not* rclone) if full tree parity is required. Sippy is the primary path because it's server-side and keeps 1:1 key parity.

---

## Phase 4 — Lightning AI Studio onboarding

1. **Account:** sign up + phone verify → free credits (~15/mo, ≈22 T4-hr, more with card-linked/onboarding bonuses). Create a Studio.
2. **Free-plan constraints:** 4 h restart cycle, persistent >=100 GB storage (survives restarts), CPU↔GPU toggle (do data prep on CPU; burn credits only on GPU).
3. **Auth:** add the wrangler credential as a Studio secret (`CLOUDFLARE_API_TOKEN`), or `npx wrangler login` in the Studio.
4. **Ingest the payload from R2 (free egress):**
   > **Studio filesystem layout (verified 2026-10-14):** `~` = `/teamspace/studios/this_studio` — Studio home, writable, **persisted across restarts** (except `node_modules`), 400 GB default disk. `/teamspace` root is Lightning-managed (no user `mkdir`); `uploads/` is web-UI-write only, read-only from Studio. S3/R2 connectors ("Add data" in UI) FUSE-mount external buckets at `/teamspace/{s3,r2}_connections/<name>` — **read-only, browsing-grade only (~600 img/s, crashes under training load)**; fine for archive browsing, do not train through FUSE. Working copy lives at **`~/catnip-data`** (Studio home; R2 stays canonical).
   ```bash
   mkdir -p ~/catnip-data
   npx wrangler r2 object get catnip-canonical/compressed/stage1_sliced.tar \
     --file ~/catnip-data/stage1_sliced.tar --remote
   tar -xf ~/catnip-data/stage1_sliced.tar -C ~/catnip-data
   npx wrangler r2 object get catnip-canonical/models/yolo26n.pt \
     --file ~/catnip-data/models/yolo26n.pt --remote
   rm ~/catnip-data/stage1_sliced.tar   # re-pull from R2 is free
   ```
   Then verify 1:1 in the Studio: `find ~/catnip-data/training/stage1_sliced/images -type f | wc -l` vs `.../labels/...` (expect 164,692 each).
5. **Env:** clone repo; `pixi install` (or venv + `ultralytics`); set `CATNIP_DATA=$HOME/catnip-data`; patch `dataset.yaml` absolute paths.
6. **Train:** `python scripts/train/stage1.py --device cuda` with `--resume` broken into chunks (free restart cycle / GPU-session cap).
7. **Checkpoint discipline (critical):** after each chunk, tar the run and push to R2 so a Studio restart never loses progress:
   ```bash
   tar -C $HOME/catnip-data -cf stage1-run.tar runs/detect/stage1
   npx wrangler r2 object put catnip-canonical/runs/stage1-run.tar \
     --file $HOME/catnip-data/stage1-run.tar --remote
   ```
   On completion, push the deliverable:
   ```bash
   npx wrangler r2 object put catnip-canonical/models/yolo26_stage1_body_head_face.pt \
     --file $HOME/catnip-data/models/yolo26_stage1_body_head_face.pt --remote
   ```
8. **(Optional) later:** pull the trained model to the HDD via `npx wrangler r2 object get catnip-canonical/models/yolo26_stage1_body_head_face.pt --file /Volumes/rifuSSD/catnip-data/models/...`.

---

## Phase 5 — Decommission GCS

### 5.1 Code/config updates (file inventory from session grep)

| File | Change |
|---|---|
| `scripts/reorganize_gcs.py` | Retire or rewrite for R2. Since wrangler has no recursive file ops, replace its gsutil tree operations with Sippy (bulk) + `wrangler r2 object put` (single artifacts) — or delete the script once GCS hygiene is done. |
| `scripts/kaggle_sync.py` | Its rclone-based `--archive-uri r2:...` / rclone-remote source path is now dead. If Kaggle stays in the loop, replace with a wrangler or S3-API archive step; otherwise remove the rclone helpers. |
| `notebooks/catnipColab.ipynb` | Replace gcsfuse/`gcloud storage` cells with `npx wrangler r2 object get` of the tarball (Colab fallback), or archive the notebook if Lightning is sole trainer. |
| `config/pipeline.yaml` | Update `paths.data` comment (gcsfuse/rclone recipes → wrangler). |
| `.env.example` | Replace Colab gcsfuse/rclone recipes with wrangler recipes; document `CLOUDFLARE_API_TOKEN`; remove the rclone R2 remote note. |
| `docs/stage1-runbook.md` | GCS layout table + sync commands → R2 (Sippy for bulk, `wrangler r2 object put/get` for artifacts). |
| `docs/architecture.md` | Storage section: GCS is archive → **R2 is archive** via wrangler/Sippy; drop "R2 migration deferred"; update egress rationale. |
| `src/preprocess/convert_labels.py:50` | Hardcoded `gs://catnip-data/data/manga/v09/0043.jpg` — leave as historical metadata or rewrite to R2 public URL; decide during PR. |
| `scripts/unify/stage1.py:19` | Comment-only (mentions gcsfuse) — optional. |

### 5.2 Label Studio dependency (pre-delete gate)
- Inventory project-2 image URLs (`gs://catnip-data/data/...`).
- If any imports still reference GCS URLs: mirror those objects to `catnip-serve` (public R2 bucket, r2.dev or `data.rifusaki.com` via `wrangler r2 bucket domain` / `dev-url`) and update the LS project, or keep GCS alive for LS reads until the annotation work is exported/complete.

### 5.3 Grace period + removal
1. After R2 verification (Phase 3.4) and ≥1 epoch trained on Lightning from R2-backed data: `npx wrangler r2 bucket sippy disable catnip-canonical`; mark GCS read-only (stop syncing to it in code).
2. Grace period ~7 days (mitigates "we forgot X looks at GCS").
3. Remove: `gcloud storage rm -r gs://catnip-data`.

### 5.4 Post-removal verification
```bash
grep -rn "gs://\|gsutil\|gcsfuse\|gcloud storage\|rclone" --include="*.py" --include="*.toml" \
     --include="*.ipynb" --include="*.md" . | grep -v "\.git/"
# expect: only intentional historical notes
```

---

## Phase 6 — Optional: Kaggle / NSFW (parked)

- Only reinstate if Kaggle GPU hours are still wanted. Then: use `nsfw_report.json` + `scan_nsfw.py` → `sanitize_names.py --exclude-*` to build a cleaned subset **from R2**, upload via `kaggle_sync.py` (with its rclone paths replaced), poll via `--check-status`.
- Otherwise: keep `nsfw_report.json` as reference; no code changes.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sippy backfill duration for 59 GB / 821k objects | Server-side (no uplink cost); reads are lazy-copied on demand; monitor `sippy get` |
| Sippy copies junk (`.DS_Store`, stale runs) | Pre-clean GCS in §3.1 before enabling |
| GCS deleted while Sippy still enabled → replicated from deleted source | `sippy disable` before GCS removal (explicitly ordered in §3.5/§5.3) |
| wrangler has no object `list`/`check` → verification gap | R2 dashboard counts; spot `object get`s; optional scoped-R2-token + `s5cmd` count as a one-off |
| Tarball (Lightning payload) stale vs current sliced set | Phase-1 gate verifies/rebuilds `compressed/stage1_sliced.tar` (dataset.yaml + 1:1 counts) |
| Lightning credits small (≈22 T4-hr/mo) | CPU for prep, GPU in focused chunks, `--resume` + R2 checkpoint tarballs so nothing is lost to the 4 h restart cycle |
| Exposed rclone R2/GCS credentials from session logs | rclone retired; rotate/revoke R2 keys; remove `[r2]`/`[gcs]` from `rclone.conf`; use fresh `CLOUDFLARE_API_TOKEN` |
| Label Studio gs:// URLs break on GCS deletion | Phase 5.2 inventory + mirror to `catnip-serve` first |
| `dataset.yaml` absolute paths break on new host | Patch paths at Lightning ingest (Phase 4.5); consider documenting a repoint step |

## Success criteria

- [ ] `npx wrangler r2 bucket sippy get catnip-canonical` shows drained/disabled after verification; R2 counts ≈ cleaned-GCS baseline
- [ ] Sliced images:labels 1:1 in R2 (train 132,435 / val 16,264 / test 15,993); `dataset.yaml` spot-check matches
- [ ] Lightning Studio pulled `stage1_sliced.tar` from R2 via wrangler and trained ≥1 epoch with `--resume` across a restart
- [ ] Trained `yolo26_stage1_body_head_face.pt` published to `r2:catnip-canonical/models/` via `wrangler r2 object put` (and pulled to HDD)
- [ ] `grep gs://` and `grep rclone` across repo = 0; GCS bucket deleted
- [ ] Monthly GCP storage/egress line gone (check next billing cycle)