import cv2, os
from pathlib import Path
from ultralytics import YOLO

# Get relative path to YOLO model (Fuyucch1/yolov8_animeface)
script_dir = Path(__file__).parent
model_path = script_dir / 'models' / 'yolov8x6_animeface.pt'
model = YOLO(str(model_path))

# Heuristic head detection
def heuristic_extraction(panel_paths, CROPS_DIR):
    """
    Extract character head candidates using blob detection and aspect ratio filtering.
    This method uses simple computer vision heuristics to find head-like shapes.
    
    Args:
        panel_paths: List of paths to extracted panel images
        CROPS_DIR: Output directory for head candidate crops
    """
    count = 0
    for p in panel_paths:
        # Load panel image
        img = cv2.imread(p)
        
        # Convert to grayscale for blob detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold: white background (240+) becomes black, content becomes white
        # This isolates dark content (characters, speech bubbles) from light background
        _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Connected component analysis to find separate blobs/regions
        # This identifies distinct objects in the binary image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        
        # Extract filename for naming output crops
        base = Path(p).stem
        
        # Analyze each connected component as a potential head candidate
        for i in range(1, num_labels):  # Skip label 0 (background)
            x,y,w,h,area = stats[i]
            
            # FILTER 1: Area threshold - ignore small noise/artifacts
            # 800 pixels ≈ 28x28 minimum for a meaningful head crop
            if area < 800:
                continue
                
            # FILTER 2: Aspect ratio - heads are roughly oval/circular
            # 0.35-1.8 range captures various head orientations and styles
            aspect = w / float(h)
            if 0.35 < aspect < 1.8:
                # Add padding around detected region for context
                # 40% width padding, 60% height padding (heads need more vertical context)
                pad_w = int(w*0.4); pad_h = int(h*0.6)
                
                # Calculate padded bounding box, ensuring we stay within image bounds
                x0 = max(0, x-pad_w); y0 = max(0, y-pad_h)
                x1 = min(img.shape[1], x+w+pad_w); y1 = min(img.shape[0], y+h+pad_h)
                
                # Extract padded crop
                crop = img[y0:y1, x0:x1]
                
                # Save crop with naming: {panel}_crop_{component_id}.jpg
                outp = os.path.join(CROPS_DIR, f"{base}_crop_{i}.jpg")
                cv2.imwrite(outp, crop)
                count += 1
    print('Saved', count, 'candidate crops to', CROPS_DIR)

# Anime face detection (Fuyucch1/yolov8_animeface)
def anime_extraction(panel_paths, CROPS_DIR, model=model):
    """
    Extract anime/manga faces using YOLOv8 trained specifically on anime faces.
    This is more accurate than heuristics but requires a specialized model.
    
    Args:
        panel_paths: List of paths to extracted panel images
        CROPS_DIR: Output directory for face crops
        model: Pre-loaded YOLO model for anime face detection
    """
    count = 0
    for p in panel_paths:
        # Run YOLO inference on the panel image
        # imgsz=512: Input image size for YOLO (larger = more accurate but slower)
        # conf=0.3: Confidence threshold (30% minimum confidence for detection)
        # iou=0.5: IoU threshold for Non-Maximum Suppression (removes overlapping boxes)
        # verbose=False: Suppress detailed output logs
        results = model.predict(p, imgsz=512, conf=0.3, iou=0.5, verbose=False)
        
        # Extract filename for naming output crops
        base = Path(p).stem
        
        # Load original image for cropping
        img = cv2.imread(p)
        
        # Process each detected face bounding box
        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.astype(int)
            
            # Crop the detected face region
            crop = img[y1:y2, x1:x2]
            
            # Skip empty crops (edge case protection)
            if crop.size == 0: 
                continue
                
            # Save crop with naming: {panel}_face_{detection_id}.jpg
            outp = os.path.join(CROPS_DIR, f"{base}_face_{i}.jpg")
            cv2.imwrite(outp, crop)
            count += 1
    print('Saved', count, 'anime face crops to', CROPS_DIR)