from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    pages_dir: Path
    panels_dir: Path
    crops_dir: Path

    curated_dataset_dir: Path
    izutsumi_dir: Path
    anti_izutsumi_dir: Path

    embed_path: Path
    crop_path: Path
    model_dir: Path

    training_dir: Path

    output_dir: Path

    img_size: int

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

def setup_dirs():
    for path in vars(settings).values():
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)

    seed_paths = [str(p) for p in sorted(settings.izutsumi_dir.glob("*.jpg"))] # Seeds
    neg_paths = [str(p) for p in sorted(settings.anti_izutsumi_dir.glob("*.jpg"))]

    return seed_paths, neg_paths