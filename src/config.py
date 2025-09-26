from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    pages_dir: Path
    panels_dir: Path
    crops_dir: Path

    curated_dataset_dir: Path
    izutsumi_dir: Path

    embed_path: Path
    crop_path: Path

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()