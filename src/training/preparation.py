import shutil, random
import os
import yaml
import logging
from src.config import settings
from pathlib import Path

logger = logging.getLogger(__name__)

# random split 80/20 for train/val
def split_data(imgs, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    return imgs[n_val:], imgs[:n_val]  # train, val


def safe_symlink(target, link_name):
    """
    Creates a symlink from link_name to target safely.
    """
    target = Path(target).resolve()
    link_name = Path(link_name)
    
    if link_name.is_symlink():
        os.unlink(link_name)

    if not link_name.exists():
        try:
            os.symlink(target, link_name)
            logger.info("Created symlink: %s -> %s", link_name, target)
        except OSError as e:
            logger.error("Failed to create symlink %s -> %s: %s", link_name, target, e)
            logger.warning("On Windows, you may need to run VS Code as Administrator or enable Developer Mode.")

def generate_training_list(images_dir, labels_dir, output_path, force_regenerate=False):
    """
    Generates a text file containing paths to images that have corresponding labels.
    """
    output_path = Path(output_path)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if output_path.exists() and not force_regenerate:
        logger.info("Found existing training list: %s", output_path)
        with open(output_path, 'r') as f:
            lines = f.readlines()
        logger.info("Loaded %d images from existing list.", len(lines))
        return output_path

    logger.info("Generating new training list: %s", output_path)
    
    image_files = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpeg"))
    logger.info("Found %d total images in '%s' directory.", len(image_files), images_dir.name)

    labeled_images = []
    unlabeled_count = 0

    for img_path in image_files:
        # construct expected label path
        try:
            rel_path = img_path.relative_to(images_dir)
            label_rel_path = rel_path.with_suffix(".txt")
            label_path = labels_dir / label_rel_path
            
            if label_path.exists():
                # use absolute path to avoid ambiguity
                labeled_images.append(str(img_path.absolute()))
            else:
                unlabeled_count += 1
        except ValueError:
            continue

    # write train list
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(labeled_images))

    logger.info("Generated %s", output_path)
    logger.info("   - labeled images (subset): %d", len(labeled_images))
    logger.info("   - unlabeled images (skipped): %d", unlabeled_count)

    if len(labeled_images) == 0:
        logger.warning("No labeled images found.")
    
    return output_path

def create_dataset_yaml(path, train_path, val_path, names, output_path="dataset.yaml"):
    """
    Creates the dataset.yaml file for YOLO training.
    """
    dataset_yaml = {
        'path': str(Path(path).resolve()),
        'train': str(Path(train_path).resolve()),
        'val': str(Path(val_path).resolve()), # using same set for val for now
        'names': names
    }

    yaml_path = Path(output_path)
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)

    logger.info("Created %s", yaml_path)
    return yaml_path

def setup_stage1_data(manga_dir=None, labels_dir=None, stage1_dir=None):
    """
    Set up Stage 1 YOLO training data (3-class: body=0, head=1, face=2):
    1. Symlink manga images with labels into stage1/images/
    2. Generate training list
    3. Generate dataset.yaml

    Class labels are read dynamically from settings.labels.stage1 (config/pipeline.yaml).
    No hardcoded class count — adding or removing classes is a config-only change.

    Args:
        manga_dir: Path to manga images. Defaults to settings.paths.manga_dir.
        labels_dir: Path to YOLO label files. Defaults to settings.paths.stage1_labels_dir.
        stage1_dir: Path to stage1 output directory. Defaults to settings.paths.stage1_dir.

    Returns:
        dict with keys: symlinks (int), train_entries (int), dataset_yaml (Path)
    """
    if manga_dir is None:
        manga_dir = settings.paths.manga_dir
    if labels_dir is None:
        labels_dir = settings.paths.stage1_labels_dir
    if stage1_dir is None:
        stage1_dir = settings.paths.stage1_dir

    if manga_dir is None or labels_dir is None or stage1_dir is None:
        missing = [k for k, v in [("manga_dir", manga_dir), ("labels_dir", labels_dir), ("stage1_dir", stage1_dir)] if v is None]
        raise ValueError(f"Required paths not configured: {', '.join(missing)}. Set them in config/pipeline.yaml or pass explicitly.")

    manga_dir = Path(manga_dir)
    labels_dir = Path(labels_dir)
    stage1_dir = Path(stage1_dir)
    images_dir = stage1_dir / "images"

    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Symlink labeled images
    logger.info("Step 1: Creating symlinks for labeled images...")
    label_files = list(labels_dir.rglob("*.txt"))
    symlink_count = 0
    for label_path in label_files:
        rel_path = label_path.relative_to(labels_dir)
        img_rel_path = rel_path.with_suffix(".jpg")
        img_src = manga_dir / img_rel_path
        img_dest = images_dir / img_rel_path

        if not img_src.exists():
            img_rel_path_png = rel_path.with_suffix(".png")
            img_src = manga_dir / img_rel_path_png
            img_dest = images_dir / img_rel_path_png

        if img_src.exists():
            img_dest.parent.mkdir(parents=True, exist_ok=True)
            safe_symlink(img_src, img_dest)
            symlink_count += 1
        else:
            logger.warning("  Image not found for %s", rel_path)

    logger.info("  Created %d symlinks in %s", symlink_count, images_dir)

    # Step 2: Generate training list
    logger.info("Step 2: Generating training list...")
    train_list_path = stage1_dir / "train.txt"
    generate_training_list(str(images_dir), str(labels_dir), str(train_list_path), force_regenerate=True)

    train_entries = 0
    if train_list_path.exists():
        with open(train_list_path, 'r') as f:
            train_entries = sum(1 for _ in f)

    # Step 3: Generate dataset.yaml
    logger.info("Step 3: Generating dataset.yaml...")
    names = {v: k for k, v in settings.labels.stage1.model_dump().items()}
    dataset_yaml_path = create_dataset_yaml(
        path=str(stage1_dir),
        train_path=str(train_list_path),
        val_path=str(train_list_path),
        names=names,
        output_path=str(stage1_dir / "dataset.yaml"),
    )

    logger.info(
        "=== Stage 1 Data Setup Complete ===\n"
        "  Images:  %s\n"
        "  Labels:  %s\n"
        "  Train list: %s\n"
        "  Dataset YAML: %s",
        images_dir, labels_dir, train_list_path, dataset_yaml_path,
    )

    return {
        "symlinks": symlink_count,
        "train_entries": train_entries,
        "dataset_yaml": dataset_yaml_path,
    }


def save_best_model(project_dir, run_name, target_dir, target_name="best.pt"):
    """
    Saves the best model from the training run to a target directory.
    """
    project_dir = Path(project_dir)
    target_dir = Path(target_dir)
    
    best_model_path = project_dir / run_name / "weights" / "best.pt"
    target_model_path = target_dir / target_name

    if best_model_path.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_path, target_model_path)
        logger.info("Model saved to %s", target_model_path)
        return target_model_path
    else:
        logger.error("Training might have failed, best.pt not found at %s", best_model_path)
        return None


def prepare_triplet_dataset(
    source_dirs: list[str | Path],
    izutsumi_dir: str | Path | None = None,
    not_izutsumi_dir: str | Path | None = None,
    val_ratio: float = 0.2,
    class_imbalance_warn_ratio: float = 5.0,
    seed: int = 42,
):
    """
    Prepares a triplet dataset from source directories containing mixed crops.

    Scans source directories for image files, classifies them by filename
    (files with "izutsumi" in path → izutsumi class, others → not-izutsumi),
    optionally adds images from pre-sorted izutsumi/not-izutsumi directories,
    splits into train/val sets, and copies to stage2_dir.

    Args:
        source_dirs: Directories with mixed crops, classified by filename pattern.
        izutsumi_dir: Additional directory with pre-sorted izutsumi images.
            Defaults to settings.paths.izutsumi_dir.
        not_izutsumi_dir: Additional directory with pre-sorted not-izutsumi images.
            Defaults to settings.paths.not_izutsumi_dir.
        val_ratio: Fraction of images reserved for validation (default 0.2).
        class_imbalance_warn_ratio: Logs WARNING if larger_class / smaller_class
            exceeds this threshold.

    Returns:
        dict with train/val per-class image counts:
        {"train": {"izutsumi": int, "not_izutsumi": int},
         "val":   {"izutsumi": int, "not_izutsumi": int}}
    """
    # Resolve defaults from settings
    if izutsumi_dir is None:
        izutsumi_dir = settings.paths.izutsumi_dir
    if not_izutsumi_dir is None:
        not_izutsumi_dir = settings.paths.not_izutsumi_dir

    izutsumi_dir = Path(izutsumi_dir)
    not_izutsumi_dir = Path(not_izutsumi_dir)
    source_dirs = [Path(d) for d in source_dirs]
    stage2_dir = settings.paths.stage2_dir

    valid_suffixes = {".jpg", ".png", ".jpeg"}

    izutsumi_images: list[Path] = []
    not_izutsumi_images: list[Path] = []

    # --- Scan source directories (filename-based classification) ---
    for src_dir in source_dirs:
        if not src_dir.exists():
            logger.warning("Source directory does not exist: %s", src_dir)
            continue

        scanned = 0
        for suffix in valid_suffixes:
            for img_path in src_dir.glob(f"*{suffix}"):
                scanned += 1
                if "izutsumi" in str(img_path).lower():
                    izutsumi_images.append(img_path)
                else:
                    not_izutsumi_images.append(img_path)

        if scanned == 0:
            logger.warning("No image files found in source directory: %s", src_dir)

    # --- Scan izutsumi-specific directory ---
    if izutsumi_dir.exists():
        for suffix in valid_suffixes:
            izutsumi_images.extend(izutsumi_dir.glob(f"*{suffix}"))
    else:
        logger.warning("Izutsumi source directory does not exist: %s", izutsumi_dir)

    # --- Scan not-izutsumi-specific directory ---
    if not_izutsumi_dir.exists():
        for suffix in valid_suffixes:
            not_izutsumi_images.extend(not_izutsumi_dir.glob(f"*{suffix}"))
    else:
        logger.warning("Not-izutsumi source directory does not exist: %s", not_izutsumi_dir)

    # --- Duplicate-free deduplication ---
    izutsumi_images = list(dict.fromkeys(izutsumi_images))
    not_izutsumi_images = list(dict.fromkeys(not_izutsumi_images))

    # --- Log collected totals ---
    logger.info(
        "Collected %d izutsumi and %d not-izutsumi images across all sources",
        len(izutsumi_images), len(not_izutsumi_images),
    )

    # Check for empty datasets
    if len(izutsumi_images) == 0:
        logger.warning("No izutsumi images found in any source directory")
    if len(not_izutsumi_images) == 0:
        logger.warning("No not-izutsumi images found in any source directory")

    # --- Split each class into train/val ---
    izutsumi_train, izutsumi_val = split_data(izutsumi_images, val_ratio, seed=seed)
    not_izutsumi_train, not_izutsumi_val = split_data(not_izutsumi_images, val_ratio, seed=seed)

    # --- Copy files to stage2_dir ---
    def _copy_images(images: list[Path], dest_dir: Path, class_name: str):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            try:
                shutil.copy(img, dest_dir / img.name)
            except (OSError, PermissionError) as e:
                logger.warning("Failed to copy %s → %s/%s: %s", img, class_name, img.name, e)

    _copy_images(izutsumi_train, stage2_dir / "train" / "izutsumi", "izutsumi")
    _copy_images(izutsumi_val,   stage2_dir / "val" / "izutsumi", "izutsumi")
    _copy_images(not_izutsumi_train, stage2_dir / "train" / "not_izutsumi", "not_izutsumi")
    _copy_images(not_izutsumi_val,   stage2_dir / "val" / "not_izutsumi", "not_izutsumi")

    # --- Build result counts ---
    result = {
        "train": {
            "izutsumi": len(izutsumi_train),
            "not_izutsumi": len(not_izutsumi_train),
        },
        "val": {
            "izutsumi": len(izutsumi_val),
            "not_izutsumi": len(not_izutsumi_val),
        },
    }

    # --- Log per-class counts for each split ---
    logger.info(
        "Train split — izutsumi: %d, not_izutsumi: %d",
        result["train"]["izutsumi"], result["train"]["not_izutsumi"],
    )
    logger.info(
        "Val split   — izutsumi: %d, not_izutsumi: %d",
        result["val"]["izutsumi"], result["val"]["not_izutsumi"],
    )

    # --- Class imbalance warning ---
    total_izutsumi = len(izutsumi_images)
    total_not_izutsumi = len(not_izutsumi_images)
    if total_izutsumi > 0 and total_not_izutsumi > 0:
        smaller = min(total_izutsumi, total_not_izutsumi)
        larger = max(total_izutsumi, total_not_izutsumi)
        ratio = larger / smaller
        if ratio > class_imbalance_warn_ratio:
            logger.warning(
                "Class imbalance detected: izutsumi=%d, not_izutsumi=%d "
                "(ratio %.1f:1, exceeds threshold %.1f)",
                total_izutsumi, total_not_izutsumi,
                ratio, class_imbalance_warn_ratio,
            )

    return result
