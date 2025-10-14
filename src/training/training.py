from src.config import settings
from ultralytics import YOLO

def build_model(path):
    return YOLO(path)

def train_model(model,
                epochs=50, 
                imgsz=settings.params.img_size, 
                batch=16, 
                lr0=1e-4, 
                freeze=10,
                workers=0,
                resume=False):
    
    # Load pretrained animeface model

    # Fine-tune on your dataset
    model.train(
        data="config/izutsumiTraining.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,       # lower LR for finetuning
        freeze=freeze,      # freeze backbone layers
        project="runs/izutsumi_finetune",
        name="exp1",
        device=settings.params.device,
        workers=workers,
        resume=resume
    )