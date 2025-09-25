import cv2, os
from pathlib import Path
from ultralytics import YOLO

# Get relative path to YOLO model (Fuyucch1/yolov8_animeface)
script_dir = Path(__file__).parent
model_path = script_dir / 'models' / 'yolov8x6_animeface.pt'
model = YOLO(str(model_path))

# Heuristic head detection
def heuristic_extraction(panel_paths, CROPS_DIR):
    count = 0
    for p in panel_paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        base = Path(p).stem
        for i in range(1, num_labels):
            x,y,w,h,area = stats[i]
            if area < 800:
                continue
            aspect = w / float(h)
            if 0.35 < aspect < 1.8:
                pad_w = int(w*0.4); pad_h = int(h*0.6)
                x0 = max(0, x-pad_w); y0 = max(0, y-pad_h)
                x1 = min(img.shape[1], x+w+pad_w); y1 = min(img.shape[0], y+h+pad_h)
                crop = img[y0:y1, x0:x1]
                outp = os.path.join(CROPS_DIR, f"{base}_crop_{i}.jpg")
                cv2.imwrite(outp, crop)
                count += 1
    print('Saved', count, 'candidate crops to', CROPS_DIR)

# Anime face detection (Fuyucch1/yolov8_animeface)
def anime_extraction(panel_paths, CROPS_DIR, model=model):
    count = 0
    for p in panel_paths:
        results = model.predict(p, imgsz=512, conf=0.3, iou=0.5, verbose=False)
        base = Path(p).stem
        img = cv2.imread(p)
        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: 
                continue
            outp = os.path.join(CROPS_DIR, f"{base}_face_{i}.jpg")
            cv2.imwrite(outp, crop)
            count += 1
    print('Saved', count, 'anime face crops to', CROPS_DIR)