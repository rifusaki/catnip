import cv2
import logging
import os
from pathlib import Path
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logger = logging.getLogger(__name__)

class SahiExtractor:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initializes the SAHI extractor with a YOLOv8 model.
        """
        self.model_path = model_path
        self.device = device
        
        logger.info("Loading YOLO model from %s onto %s...", model_path, device)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=self.model_path,
            confidence_threshold=0.3,
            device=self.device, # "mps", "cuda", or "cpu"
        )

    def extract_from_directory(self, input_dir: str, output_dir: str, slice_height=640, slice_width=640, overlap_ratio=0.2):
        """
        Reads all images in input_dir, performs sliced inference, and saves the cropped bounding boxes to output_dir.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))
        
        logger.info("Found %d images to process.", len(image_paths))
        crop_count = 0
        
        for img_path in tqdm(image_paths, desc="Extracting via SAHI"):
            # Run SAHI prediction
            result = get_sliced_prediction(
                str(img_path),
                self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                perform_standard_pred=True, # Global + Local context
                postprocess_type="NMM", # Non-Maximum Merging handles overlaps best
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5
            )
            
            # Read original image to extract crops
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Iterate through merged bounding boxes
            for i, object_prediction in enumerate(result.object_prediction_list):
                bbox = object_prediction.bbox
                x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
                
                # Extract crop
                crop = img[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Save crop: {original_filename}_crop_{i}.jpg
                    save_name = f"{img_path.stem}_crop_{i}.jpg"
                    save_path = output_dir / save_name
                    cv2.imwrite(str(save_path), crop)
                    crop_count += 1
                    
        logger.info("Extraction complete! Saved %d character crops to %s", crop_count, output_dir)
        return crop_count

# Example usage (will be called from a main.py later)
if __name__ == "__main__":
    # extractor = SahiExtractor(model_path="models/yolov8x6_animeface.pt", device="mps")
    # extractor.extract_from_directory("data/raw_pages", "data/crops")
    pass
