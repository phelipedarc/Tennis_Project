import os, sys
GPU = '0,4,5,6'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["WORLD_SIZE"] = "1"


import ultralytics
from ultralytics import YOLO
from PIL import Image


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.device_count()

#Hyper_Parameters:

PATIENCE = 200
PROJECT_NAME = 'YOLOv10s_1008'
BATCH = 128
TRAIN_yaml = "/tf/astrodados/TennisPlayerDetection_dataset/NEW/newdataset.yaml"
EPOCHS = 1000
# Print hyperparameters for user confirmation
print("\nTraining Configuration:")
print(f" - Patience: {PATIENCE}")
print(f" - Project Name: {PROJECT_NAME}")
print(f" - Batch Size: {BATCH}")
print(f" - Training Dataset: {TRAIN_yaml}")
print(f" - Epochs: {EPOCHS}\n")

# Initialize model
print("Loading YOLO model...")
model = YOLO("yolov10s.pt")
print("Model loaded successfully!\n")

print("\nStarting training process...")
print(f"Training on GPUs: {GPU}")

results = model.train(data=TRAIN_yaml, 
                      epochs=EPOCHS,
                      batch=BATCH, 
                      imgsz=640,
                      device=GPU, 
                      workers=80, 
                      patience=PATIENCE,
                      single_cls=False,
                      multi_scale=True, 
                      project=PROJECT_NAME)
print("\nTraining completed successfully!")
#model.val()
# Export model and notify user
print("\nExporting the trained model...")
model.export()




