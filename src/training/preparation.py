import shutil, random
from src.config import settings
from pathlib import Path

# random split 80/20 for both classes
def split_data(imgs, val_ratio=0.2):
    random.shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    return imgs[n_val:], imgs[:n_val]  # train, val


def copy_11(out_dir, imgs, split, class_id):
    if class_id == 0: type = 'notIzutsumi'
    elif class_id == 1: type = 'izutsumi'
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


    print(f'Data prepared in {out_dir}')