from pathlib import Path
from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Any

class Paths(BaseModel):
    data: Path
    
    # Training paths (GCS/Colab)
    manga_dir: Path | None = None
    annotations_dir: Path | None = None
    
    # Inference/Processing paths
    pages_dir: Path | None = None
    panels_dir: Path | None = None
    crops_dir: Path | None = None
    izutsumi_dir: Path | None = None
    not_izutsumi_dir: Path | None = None
    embs_dir: Path | None = None
    training_dir: Path | None = None
    
    model_dir: Path
    output_dir: Path
    runs_dir: Path

class Params(BaseModel):
    img_size: int
    device: str

class Settings(BaseModel):
    paths: Paths
    params: Params


def load_settings(path: str | Path = "config/pipeline.yaml") -> Settings:
    root = Path.cwd()
    config_path = root / path
    cfg = OmegaConf.load(config_path)
    
    # resolve all ${...} interpolations
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return Settings(**resolved)

settings = load_settings()


def setup_dirs():
    for path in vars(settings).values():
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)

    izutsumi = [str(p) for p in sorted(settings.paths.izutsumi_dir.glob("*.jpg"))] # Seeds
    notIzutsumi = [str(p) for p in sorted(settings.paths.not_izutsumi_dir.glob("*.jpg"))]

    return izutsumi, notIzutsumi