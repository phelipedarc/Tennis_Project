import json
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil
import os
#Choose the videos ID and the DATASET base directory
#rayan:
# Treino: '28','41','34','21','50','2','38','47'
# Validação: 39,47,5
# Teste: 30 e 45

#Darc:
# Treino: '28','41','34','21','50','2','38','47'
# Validação: 39,5
# Teste: 30 e 45

base_dataset = '/tf/astrodados/TennisPlayerDetection_dataset/NEW/'

unique_vid_ids = set()
file_paths = glob.glob(f'/tf/astrodados/phelipedata/YOLO_bboxes_dataset/dataset/labels/train/*')
# Loop through the file paths and extract vid_ids
for path in file_paths:
    # Get the filename from the path
    filename = os.path.basename(path)
    
    # Split the filename to extract the vid_id (it starts with 'vid_')
    vid_id = filename.split('_')[1]  # Extract 'vid_XX' part
    unique_vid_ids.add(vid_id)

# Output the number of unique vid_ids
print(f"Number of unique vid_ids: {len(unique_vid_ids)}")
print(f"Unique vid_ids: {unique_vid_ids}")

#{'x30', '28', '41', 'x5', '34', '21', 'x45', '50', 'x47', 'x39', '2', '38'}
#'35', '33', '50', '29', '25', '41', '5', '36', '45', '44', '39', '37', '21', '38', '24', '28', '30', '34', '2', '47'
#create empty ID arrays if you want only to delete the content of the Directories
id_train = ['35', '33', '50', '29', '25', '41', '36', '44', '37', '21', '24', '28', '34', '2']
id_validation=['39','5','47']
id_test=['30','38','45']
print(f'Train ID: {id_train}',f'Validation ID: {id_validation}', f'Test ID: {id_test}')

def create_yolo_folder_structure(base_dir=base_dataset, clear=False):
    # Define the folder names for the dataset
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val', 'images/test', 'labels/test']
    
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory: {base_dir}")
    
    # Loop through subdirectories
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        
        # Clear the directory if clear=True
        if clear and os.path.exists(path):
            shutil.rmtree(path)  # Remove the directory and its contents
            print(f"Cleared directory: {path}")
        
        # Create the subdirectory (or recreate it if it was cleared)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

# Call the function to create the YOLO folder structure
create_yolo_folder_structure(clear=True)  # Set clear=True to clear the folders before creating them
print('='*90)

for id_vid in id_train:
    print('#'*90)
    print(f'Copying the TRAIN Dataset - VidID:{id_vid}')
    path_images = glob.glob(f'/tf/astrodados/phelipedata/YOLO_bboxes_dataset/dataset/images/train/vid_{id_vid}_*')
    path_labels=glob.glob(f'/tf/astrodados/phelipedata/YOLO_bboxes_dataset/dataset/labels/train/vid_{id_vid}_*')
    path_images.sort()
    path_labels.sort()
    if len(path_images)==len(path_labels):
        for j in tqdm(range(len(path_images))):
            #Copying the images
            src = path_images[j]
            dest = os.path.join(base_dataset,'images/train')
            shutil.copy2(src, dest)
            #Copying the labels
            src0 = path_labels[j]
            dest0 = os.path.join(base_dataset,'labels/train')
            shutil.copy2(src0, dest0)
    else:  
        print('Error - Number of images ≠ Number of labels')


print('='*90)
# print('*'*90)
# print('='*90)
for id_vid in id_validation:
    print('#'*90)
    print(f'Copying the VALIDATION Dataset - VidID:{id_vid}')
    path_images = glob.glob(f'/tf/astrodados/phelipedata/YOLO_bboxes_dataset/dataset/images/train/vid_{id_vid}_*')
    path_labels=glob.glob(f'/tf/astrodados/phelipedata/YOLO_bboxes_dataset/dataset/labels/train/vid_{id_vid}_*')
    path_images.sort()
    path_labels.sort()
    if len(path_images)==len(path_labels):
        for j in tqdm(range(len(path_images))):
            #Copying the images
            src = path_images[j]
            dest = os.path.join(base_dataset,'images/val')
            shutil.copy2(src, dest)
            #Copying the labels
            src0 = path_labels[j]
            dest0 = os.path.join(base_dataset,'labels/val')
            shutil.copy2(src0, dest0)
    else:  
        print('Error - Number of images ≠ Number of labels')


print('='*90)
# print('*'*90)
# print('='*90)
for id_vid in id_test:
    print('#'*90)
    print(f'Copying the TEST Dataset - VidID:{id_vid}')
    path_images = glob.glob(f'/tf/astrodados/phelipedata/YOLO_bboxes_dataset/dataset/images/train/vid_{id_vid}_*')
    path_labels=glob.glob(f'/tf/astrodados/phelipedata/YOLO_bboxes_dataset/dataset/labels/train/vid_{id_vid}_*')
    path_images.sort()
    path_labels.sort()
    if len(path_images)==len(path_labels):
        for j in tqdm(range(len(path_images))):
            #Copying the images
            src = path_images[j]
            dest = os.path.join(base_dataset,'images/test')
            shutil.copy2(src, dest)
            #Copying the labels
            src0 = path_labels[j]
            dest0 = os.path.join(base_dataset,'labels/test')
            shutil.copy2(src0, dest0)
    else:  
        print('Error - Number of images ≠ Number of labels')

#############################################################################################
train = glob.glob(f'/tf/astrodados/TennisPlayerDetection_dataset/NEW/images/train/*')
valid = glob.glob(f'/tf/astrodados/TennisPlayerDetection_dataset/NEW/images/val/*')
test = glob.glob(f'/tf/astrodados/TennisPlayerDetection_dataset/NEW/images/test/*')
total = len(train) + len(valid) + len(test)
print(f'Training size: {len(train)} -- Val size: {len(valid)} -- Test size: {len(test)}')
print(f'Training size: {len(train)/total} -- Val size: {len(valid)/total} -- Test size: {len(test)/total}')