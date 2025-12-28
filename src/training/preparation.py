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


def prepare_data(izutsumiPaths, notIzutsumiPaths, out_dir = settings.paths.training_dir, version = 0):
    # create YOLO folder structure
    for split in ["train", "val"]:
        if version == 8:
            for sub in ["images", "labels"]:
                (out_dir / split / sub).mkdir(parents=True, exist_ok=True)

    print(f"Izutsumi: {len(izutsumiPaths)} | Not Izutsumi: {len(notIzutsumiPaths)}")

    izutsumiPaths, notIzutsumiPaths = ([Path(i) for i in izutsumiPaths], [Path(i) for i in notIzutsumiPaths])

    iz_train, iz_val = split_data(izutsumiPaths)
    not_train, not_val = split_data(notIzutsumiPaths)

    if version == 8:
        copy_and_label_v8(out_dir, iz_train, "train", 0)
        copy_and_label_v8(out_dir, iz_val, "val", 0)
        copy_and_label_v8(out_dir, not_train, "train", 1)
        copy_and_label_v8(out_dir, not_val, "val", 1)
    elif version == 11:
        copy_11(out_dir, iz_train, "train", 0)
        copy_11(out_dir, iz_val, "val", 0)
        copy_11(out_dir, not_train, "train", 1)
        copy_11(out_dir, not_val, "val", 1)
        print('not implemented xd')


def safe_symlink(target, link_name):
    """
    Creates a symlink from link_name to target safely.
    """
    target = Path(target)
    link_name = Path(link_name)
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
        'path': str(path),
        'train': str(train_path),
        'val': str(train_path), # using same set for val for now
        'names': names
    }

    yaml_path = Path(output_path)
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)

    print(f"created {yaml_path}")
    return yaml_path

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




    print(f'Data prepared in {out_dir}')