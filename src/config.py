from pathlib import Path
from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Any

class Paths(BaseModel):
    data: Path
    pages_dir: Path
    panels_dir: Path
    crops_dir: Path
    izutsumi_dir: Path
    not_izutsumi_dir: Path
    embs_dir: Path
    training_dir: Path
    model_dir: Path
    output_dir: Path

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