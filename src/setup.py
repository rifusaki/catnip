from pathlib import Path
from .config import settings

def setup_dirs():
    for path in vars(settings).values():
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)

    seed_paths = [str(p) for p in sorted(settings.paths.izutsumi_dir.glob("*.jpg"))] # Seeds
    neg_paths = [str(p) for p in sorted(settings.paths.not_izutsumi_dir.glob("*.jpg"))]

    return seed_paths, neg_paths