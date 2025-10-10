import cv2, os, numpy as np
from pathlib import Path

# Hybrid panel extraction: contours + Hough-based refinement
def hybrid_extract_panels(page_path, out_dir, min_area=20000):
    """
    Extract manga panels from a page using hybrid approach:
    1. Contour detection to find panel boundaries
    2. Hough line detection to split panels along gutters (whitespace)
    
    Args:
        page_path: Path to manga page image
        out_dir: Output directory for extracted panels
        min_area: Minimum area threshold to filter noise (default: 20000 pixels)
    """
    # Load and validate input image
    img = cv2.imread(page_path)
    if img is None:
        return 0
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold: white background (240+) becomes black, content becomes white
    # This inverts typical manga where panels are darker than gutters
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find external contours (panel boundaries) in the binary image
    # RETR_EXTERNAL only gets outermost contours, ignoring internal details
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract filename stem for naming output files
    base = Path(page_path).stem
    saved = 0

    # Process each detected contour as a potential panel
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter out small noise/artifacts based on area threshold
        if area < min_area:
            continue
            
        # Get bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(cnt)
        panel = img[y:y+h, x:x+w]

        # REFINEMENT PHASE: Use Hough line detection to split panels along gutters
        # Convert panel to grayscale for edge detection
        g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection to find strong edges (potential gutter boundaries)
        edges = cv2.Canny(g, 50, 150, apertureSize=3)
        
        # Hough line detection to find straight lines (gutters between sub-panels)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=200,
                                minLineLength=int(0.5*min(panel.shape[:2])),  # Min length = 50% of smaller dimension
                                maxLineGap=20)  # Allow 20px gaps in lines

        if lines is not None:
            # Process detected lines to split the panel
            # Split along detected horizontal/vertical lines
            masks = []
            for line in lines:
                x1,y1,x2,y2 = line[0]
                
                # Filter for mostly horizontal or vertical lines (gutters)
                # Tolerance of 20px for slight imperfections
                if abs(x2-x1) < 20 or abs(y2-y1) < 20:  # vertical or horizontal
                    # Create a mask for this gutter line
                    mask = np.zeros_like(g)
                    cv2.line(mask, (x1,y1), (x2,y2), 255, thickness=10)  # Thick line to ensure separation
                    masks.append(mask)
                    
            if masks:
                # Combine all gutter masks into one
                combined = np.zeros_like(g)
                for m in masks: 
                    combined = cv2.bitwise_or(combined, m)
                    
                # Invert mask: gutters become black, panels become white
                inv = cv2.bitwise_not(combined)
                
                # Find connected components (individual sub-panels) after gutter removal
                segs, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Save each sub-panel separately
                for j, s in enumerate(segs):
                    xx,yy,ww,hh = cv2.boundingRect(s)
                    
                    # Only save if sub-panel is large enough (half of min_area)
                    if ww*hh > min_area/2:
                        crop = panel[yy:yy+hh, xx:xx+ww]
                        # Name: {page}_panel_{contour_id}_{sub_panel_id}.jpg
                        out_path = os.path.join(out_dir, f"{base}_panel_{i}_{j}.jpg")
                        cv2.imwrite(out_path, crop)
                        saved += 1
                continue  # Skip to next contour (this panel was successfully split)

        continue  # Skip to next contour (this panel was successfully split)
    
        # FALLBACK: If no useful gutter lines detected, save the entire contour as one panel
        # This handles simple panels without internal gutters
        out_path = os.path.join(out_dir, f"{base}_panel_{i}.jpg")
        cv2.imwrite(out_path, panel)
        saved += 1
    return saved

def extract_panels(page_paths, PANELS_DIR):
    """
    Process multiple manga pages for panel extraction.
    
    Args:
        page_paths: List of paths to manga page images
        PANELS_DIR: Output directory for extracted panels
    """
    count = 0
    # Process each page sequentially
    for p in page_paths:
        count += hybrid_extract_panels(p, PANELS_DIR)
    print('Saved', count, 'panels to', PANELS_DIR)