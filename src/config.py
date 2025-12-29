from pathlib import Path
from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import Any

class Paths(BaseModel):
    data: Path
    
    manga_dir: Path | None = None
    annotations_dir: Path | None = None
    ls_exports_dir: Path | None = None
    model_dir: Path | None = None
    output_dir: Path | None = None
    runs_dir: Path | None = None

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