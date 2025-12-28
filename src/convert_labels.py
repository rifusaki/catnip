import json
import os
from pathlib import Path

def convert_label_studio_to_yolo(json_path, output_dir, class_map):
    """
    Converts Label Studio JSON export to YOLO format txt files.
    
    Args:
        json_path (str): Path to the Label Studio JSON export.
        output_dir (str): Root directory to save YOLO labels.
        class_map (dict): Mapping from label name to class ID.
    """
    with open(json_path, 'r') as f:
        tasks = json.load(f)
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for task in tasks:
        # 1. Extract relative path from URL
        # URL format: gs://catnip-data/manga/v09/0001.jpg
        data_dict = task['data']
        url = data_dict.get('url') or data_dict.get('image')
        
        if not url:
             print(f"Skipping task {task.get('id')}: No 'url' or 'image' found in data.")
             continue

        if "manga/" in url:
            # Extract everything after 'manga/'
            rel_path = url.split("manga/")[-1]
        else:
            print(f"Skipping URL with unexpected format: {url}")
            continue
            
        # 2. Determine output path
        # v09/0001.jpg -> v09/0001.txt
        image_rel_path = Path(rel_path)
        label_rel_path = image_rel_path.with_suffix('.txt')
        label_full_path = output_dir / label_rel_path
        
        # Create parent directories if needed
        label_full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 3. Process annotations
        yolo_lines = []
        if 'annotations' in task:
            for annotation in task['annotations']:
                if 'result' in annotation:
                    for result in annotation['result']:
                        if 'type' in result and result['type'] == 'rectanglelabels':
                            value = result['value']
                            labels = value.get('rectanglelabels', [])
                            
                            if not labels:
                                continue
                                
                            # Assuming single label per box for now, or take the first one
                            label_name = labels[0]
                            if label_name not in class_map:
                                print(f"Warning: Unknown label '{label_name}' in {rel_path}. Skipping.")
                                continue
                                
                            class_id = class_map[label_name]
                            
                            # Label Studio (0-100) to YOLO (0-1)
                            # LS: x, y (top-left), width, height
                            # YOLO: x_center, y_center, width, height
                            
                            x = value['x']
                            y = value['y']
                            w = value['width']
                            h = value['height']
                            
                            x_center = (x + w / 2) / 100.0
                            y_center = (y + h / 2) / 100.0
                            w_norm = w / 100.0
                            h_norm = h / 100.0
                            
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 4. Write to file (only if there are labels? or empty file for negatives?)
        # Usually empty file is good for negatives in YOLO
        with open(label_full_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            
        count += 1
        
    print(f"Processed {count} tasks. Labels saved to {output_dir}")

if __name__ == "__main__":
    # Configuration
    JSON_FILE = "project-1-at-2025-12-26-16-22-05d8afa7.json"  # Update if needed
    OUTPUT_DIR = "data/annotations"
    
    # Define your class mapping here
    CLASS_MAP = {
        "izutsumi": 0,
        "izutsumi_face": 1
    }
    
    if os.path.exists(JSON_FILE):
        convert_label_studio_to_yolo(JSON_FILE, OUTPUT_DIR, CLASS_MAP)
    else:
        print(f"File {JSON_FILE} not found.")
