import os
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

# Set up environment variables and device
GPU = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
os.environ['OMP_NUM_THREADS'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.device_count()



def main():
    """
    Main function that sets up video paths, YOLO model, and initiates the saving process.

    This is the entry point of the script.
    """
    # Input variables
    date = '1008'
    video_list = ['/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_22_0Ti_2yT1Gng_Pos.mp4',
                  '/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_31_t2weJQo-Z2s_Pos.mp4',
                 '/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_32_KBZ8zKpQFSY_Pos.mp4']
        # video_list = ['/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_20_iHJ1jdvLQAU_Pos.mp4',
    #               '/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_23_ACYPajC8S5c_Pos.mp4',
    #               '/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_22_0Ti_2yT1Gng_Pos.mp4',
    #               '/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_31_t2weJQo-Z2s_Pos.mp4',
    #              '/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_32_KBZ8zKpQFSY_Pos.mp4']

    #video_list = ['/tf/astrodados/phelipedata/YOLOV8/VIDEOS_Rayan/vid_7_OG_FULL_CLEAN_1080p.mp4']
    
    model_yolo = f"/tf/astrodados/phelipedata/YOLOV8/Player_detection/YOLOv10s_{date}/train/weights/best.pt"

    # Run YOLO inference and save results
    save_video_yolo(video_list, model_yolo)





def move_and_rename_directory(source_path, target_dir):
    """
    Moves a directory from source_path to target_dir and renames it.

    Parameters:
    - source_path (Path): The current path of the directory.
    - target_dir (Path): The target path where the directory will be moved.

    Returns:
    - None
    """
    try:
        # Move the directory to the target directory
        shutil.move(str(source_path), str(target_dir))
        print(f"Directory moved from {source_path} to {target_dir}.")
    except Exception as e:
        print(f"Error while moving the directory: {e}")

def save_video_yolo(videos_path, model_path):
    """
    Runs YOLO model predictions on a list of videos and moves the results to a new directory.

    Parameters:
    - videos_path (list): List of video paths to run inference on.
    - model_path (str): Path to the YOLO model.

    Returns:
    - None
    """
    for i in range(len(videos_path)):
        print(f'SAVING VIDEO: {videos_path[i]}')
        print('#' * 90)

        # Load YOLO model
        model = YOLO(model_path).to('cuda:0')

        # Run inference and save results
        result = model.predict(source=videos_path[i], save=True, device='cuda:0', save_txt=True)

        # Extract necessary paths for renaming and moving the directory
        destination_path = Path('/tf/astrodados/')  # Destination for the results
        oldname_dir = Path(result[0].save_dir.split('/')[-1])  # Old directory name
        newname_dir = Path(result[0].path.split('/')[-1].split('vid_')[-1].split('.')[0])  # New directory name

        # Define source and target paths
        source_path = Path(str(result[0].save_dir))  # Directory where YOLO saves results
        target_dir = destination_path / newname_dir  # New destination and name

        print(f'FILES SAVED: {source_path}')
        
        # Move and rename the directory
        move_and_rename_directory(source_path, target_dir)

    print('Inference Saved')


if __name__ == '__main__':
    main()












