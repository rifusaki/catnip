# Catnip

> A work in progress

Catnip is a small Jupyter notebook for those obsessive lovers of a specific manga character with spare time and computing power.

Input some manga pages, add some seeds, and rejoice.

## Usage
This is the very inefficient way in which I'm testing this. So far.

- I suggest using [Pixi](https://pixi.sh/latest/) for package management
- Set up a directory with sample manga pages where the desired character appears
- Run panel and facial cropping cells
- Collect samples of the desired character in a separate directory
- Run panel/facial cropping cells for the rest of the manga and wait
- Point to seeds (samples) directory, run the nearest neighbor algorithm and wait (again)

### Comments
Adenzu's Panel Extractor algorithm is excellent but takes a long time to run, even without split-cell. [panelExtraction.py](src/panelExtraction.py) includes a faster albeit more basic and somewhat less accurate approach. The stripped-down version used here was packaged by [avan06](https://github.com/avan06/adenzu-manga-panel-extractor-src) and an AI-generated API doc is available in [this file](docs/api.md).

### Credits
- [adenzu/Manga-Panel-extractor](https://github.com/adenzu/Manga-Panel-Extractor) and its packaged version [avan06/adenzu-manga-panel-extractor-src](https://github.com/avan06/adenzu-manga-panel-extractor-src)
- [Fuyucch1/yolov8_animeface](https://github.com/Fuyucch1/yolov8_animeface/tree/main?tab=readme-ov-file)