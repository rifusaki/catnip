# Catnip

> i do love izutsumi

The goal of this project is to answer the age-old question of "is Izutsumi in this image?"

![izutsumi](https://data.izutsumi.art/manga/SmartSelect_20240620_141333_Gallery.webp)

## Architecture

I am using a two-stage pipeline: detect (object bounding) and identify (classification).

## Datasets

### izutsumi

Own dataset. I used Label Studio to manually annotate bounding boxes on Dungeon Meshi manga pages. The labels used allow for the usage of the same dataset in both stages.

### [manga109](https://huggingface.co/datasets/hal-utokyo/Manga109)

Each volume of manga corresponds to a single XML file. The XML files are UTF-8 encoded and structured in the following way.

```md
* **`book`** Title of the work
    * **`characters`** List of characters appearing in the work
        * **`character`**
        * **`character`**
        * ...
    * **`pages`** List of pages in double-page spread manner
        * **`page`**
        * **`page`**
        * ...
```

`character` gives the correspondence between each character's name and its ID. `page` gives an overview (page number, image size) and the objects' information in the page. There are four types of objects; each has an object ID, rectangular area (xmin, xmax, ymin, ymax), and additional information specific to the object type. All IDs are 8-digit hexadecimal numbers and are unique throughout this dataset.

- 01 `face`: Face of a character.
  - > Character ID is attached.
-  02 `body`: Body of a character.
   -  > Character ID is attached.
-  03 `text`: Typed text and some handwritten text.
   - > Text content is recorded.
 - 04 `frame`: Frame of a panel.
   - > No additional information. 

I used [manga109api](https://github.com/manga109/manga109api) to parse and normalize this dataset.

### [deepghs/anime_head_detection](https://huggingface.co/datasets/deepghs/anime_head_detection)

We have three datasets:

#### v1.0 (YOLOv8)

High-quality manually annotated head detection data. Includes images sampled from gelbooru and fancaps. All heads with human characteristics (including front, side, and back views) are marked, while non-human heads without significant humanoid features are not considered as "heads".

#### v2.0 (YOLOv8)

High-quality manually annotated head detection dataset, containing images from bangumi and a large number of complex multi-person illustrations, which can significantly improve the training quality of the v1.0 dataset. Recommended to be used together with v1.0 dataset.

#### ani_face_detection (YOLOv8)

A high-quality third-party dataset that can be used for training directly. Although its name includes `face`, but what it actually annotated are `head`.

### [nyuuzyou/AnimeHeadsv3](https://huggingface.co/datasets/nyuuzyou/AnimeHeadsv3) (COCO, augmented)

The dataset with augmentation has the following preprocessing parameters:

```md
Resize: Fit within 640x640
```

The following augmentation parameters were applied to the dataset:

```md
Outputs per training example: 3
Flip: Horizontal
Saturation: Between -40% and +40%
Blur: Up to 4px
Noise: Up to 4% of pixels
```
