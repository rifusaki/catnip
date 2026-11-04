# Headless 24/7 Colab Training Runner

This setup runs Google Colab L4 GPU training driven by your 24/7 home server (`ssh 360 wsl`), preventing laptop sleep disconnects.

## Quick Start on `360 wsl`

1. **SSH to 360 WSL:**
   ```bash
   ssh 360 "wsl"
   ```

2. **Navigate to the colab directory:**
   ```bash
   cd ~/catnip-colab
   ```

3. **Add your R2 secrets:**
   ```bash
   nano .env
   ```
   Paste your `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.

4. **Launch inside `tmux` (24/7 background session):**
   ```bash
   tmux new -s catnip-training
   ./launch_360.sh
   ```
   Press `Ctrl+B`, then `D` to detach. You can close your laptop, turn off Wi-Fi, or go to sleep.

5. **Reattach anytime to view live progress:**
   ```bash
   ssh 360 "wsl tmux attach -t catnip-training"
   ```

## Automatic Resume
The runner checks Cloudflare R2 for `runs/stage1-last.pt`. If present, it automatically downloads it and resumes training (e.g. from Epoch 4) with zero manual intervention.
Checkpoints continue syncing to R2 every 10 minutes.
