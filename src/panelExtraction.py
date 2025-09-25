import cv2, os, numpy as np
from pathlib import Path

# Hybrid panel extraction: contours + Hough-based refinement
def hybrid_extract_panels(page_path, out_dir, min_area=20000):
    img = cv2.imread(page_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Contour detection
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base = Path(page_path).stem
    saved = 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        panel = img[y:y+h, x:x+w]

        # Refine with Hough line detection (split by gutters)
        g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=200,
                                minLineLength=int(0.5*min(panel.shape[:2])),
                                maxLineGap=20)

        if lines is not None:
            # Split along detected horizontal/vertical lines
            masks = []
            for line in lines:
                x1,y1,x2,y2 = line[0]
                if abs(x2-x1) < 20 or abs(y2-y1) < 20:  # vertical or horizontal
                    mask = np.zeros_like(g)
                    cv2.line(mask, (x1,y1), (x2,y2), 255, thickness=10)
                    masks.append(mask)
            if masks:
                combined = np.zeros_like(g)
                for m in masks: combined = cv2.bitwise_or(combined, m)
                inv = cv2.bitwise_not(combined)
                segs, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for j, s in enumerate(segs):
                    xx,yy,ww,hh = cv2.boundingRect(s)
                    if ww*hh > min_area/2:
                        crop = panel[yy:yy+hh, xx:xx+ww]
                        out_path = os.path.join(out_dir, f"{base}_panel_{i}_{j}.jpg")
                        cv2.imwrite(out_path, crop)
                        saved += 1
                continue

        # If no useful lines, just save the contour box
        out_path = os.path.join(out_dir, f"{base}_panel_{i}.jpg")
        cv2.imwrite(out_path, panel)
        saved += 1
    return saved

def extract_panels(page_paths, PANELS_DIR, method):
    count = 0
    for p in page_paths:
        count += hybrid_extract_panels(p, PANELS_DIR)
    print('Saved', count, 'panels to', PANELS_DIR)