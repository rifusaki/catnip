import shutil, random
import os
import yaml
from src.config import settings
from pathlib import Path

# random split 80/20 for both classes
def split_data(imgs, val_ratio=0.2):
    random.shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    return imgs[n_val:], imgs[:n_val]  # train, val


def copy_11(out_dir, imgs, split, class_id):
    if class_id == 0: type = 'izutsumi'
    elif class_id == 1: type = 'notIzutsumi'
    (out_dir / split / type ).mkdir(parents=True, exist_ok=True)
    for img in imgs:
        dest_img = out_dir / split / type / img.name
        shutil.copy(img, dest_img)


# copy images and create YOLO labels
def copy_and_label_v8(out_dir, imgs, split, class_id):

    for img in imgs:
        dest_img = out_dir / split / "images" / img.name
        shutil.copy(img, dest_img)

        # create empty label box
        label_path = out_dir / split / "labels" / (img.stem + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")



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
            print(f"Created symlink: {link_name} -> {target}")
        except OSError as e:
            print(f"Failed to create symlink {link_name} -> {target}: {e}")
            print("On Windows, you may need to run VS Code as Administrator or enable Developer Mode.")

def generate_training_list(images_dir, labels_dir, output_path, force_regenerate=False):
    """
    Generates a text file containing paths to images that have corresponding labels.
    """
    output_path = Path(output_path)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if output_path.exists() and not force_regenerate:
        print(f"found existing training list: {output_path}")
        with open(output_path, 'r') as f:
            lines = f.readlines()
        print(f"loaded {len(lines)} images from existing list.")
        return output_path

    print(f"generating new training list: {output_path}")
    
    image_files = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpeg"))
    print(f"found {len(image_files)} total images in '{images_dir.name}' directory.")

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

    print(f"generated {output_path}")
    print(f"   - labeled images (subset): {len(labeled_images)}")
    print(f"   - unlabeled images (skipped): {unlabeled_count}")

    if len(labeled_images) == 0:
        print("warning: no labeled images found.")
    
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

    print(f"created {yaml_path}")
    return yaml_path

def setup_stage1_data(manga_dir=None, labels_dir=None, stage1_dir=None):
    """
    Set up Stage 1 YOLO training data:
    1. Symlink manga images with labels into stage1/images/
    2. Generate training list
    3. Generate dataset.yaml

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
    print("Step 1: Creating symlinks for labeled images...")
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
            print(f"  Warning: Image not found for {rel_path}")

    print(f"  Created {symlink_count} symlinks in {images_dir}")

    # Step 2: Generate training list
    print("\nStep 2: Generating training list...")
    train_list_path = stage1_dir / "train.txt"
    generate_training_list(str(images_dir), str(labels_dir), str(train_list_path), force_regenerate=True)

    train_entries = 0
    if train_list_path.exists():
        with open(train_list_path, 'r') as f:
            train_entries = sum(1 for _ in f)

    # Step 3: Generate dataset.yaml
    print("\nStep 3: Generating dataset.yaml...")
    names = {v: k for k, v in settings.labels.stage1.model_dump().items()}
    dataset_yaml_path = create_dataset_yaml(
        path=str(stage1_dir),
        train_path=str(train_list_path),
        val_path=str(train_list_path),
        names=names,
        output_path=str(stage1_dir / "dataset.yaml"),
    )

    print(f"\n=== Stage 1 Data Setup Complete ===")
    print(f"  Images:  {images_dir}")
    print(f"  Labels:  {labels_dir}")
    print(f"  Train list: {train_list_path}")
    print(f"  Dataset YAML: {dataset_yaml_path}")

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
        print(f"Model saved to {target_model_path}")
        return target_model_path
    else:
        print(f"Training might have failed, best.pt not found at {best_model_path}")
        return None



