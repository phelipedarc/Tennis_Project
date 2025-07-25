from moviepy.editor import VideoFileClip
import os
import glob

def convert_avi_to_mp4(input_file):
    # Ensure the input file has an .avi extension
    if not input_file.endswith('.avi'):
        print("Input file is not an .avi file")
        return
    
    # Define the output file path (replace .avi with .mp4)
    output_file = os.path.splitext(input_file)[0] + '.mp4'
    
    # Load the video file
    video_clip = VideoFileClip(input_file)
    
    # Write the output video in MP4 format
    video_clip.write_videofile(output_file, codec='libx264')
    
    print(f"Converted {input_file} to {output_file}")


# input_file = ['/tf/astrodados/30_413ultKeMq4_Pos/vid_30_413ultKeMq4_Pos.avi',
#                   '/tf/astrodados/38_JSo7ywmIfrk_Pos/vid_38_JSo7ywmIfrk_Pos.avi',
#                   '/tf/astrodados/45_SActabD3zsU_Pos/vid_45_SActabD3zsU_Pos.avi',
#                   '/tf/astrodados/47_u_l8h2vqTYc_Pos/vid_47_u_l8h2vqTYc_Pos.avi']

# input_file = ['/tf/astrodados/7_OG_FULL_CLEAN_1080p/vid_7_OG_FULL_CLEAN_1080p.avi']
input_file = glob.glob('/tf/astrodados/*/*.avi')

#/tf/astrodados/30_413ultKeMq4_Pos/vid_30_413ultKeMq4_Pos.avi
for i in range(len(input_file)):
    convert_avi_to_mp4(input_file[i])