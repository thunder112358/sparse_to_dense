#!/usr/bin/env python3
"""
Video denoising using optical flow and temporal averaging.
This script uses FALDOI optical flow estimation to align frames
and performs temporal averaging for denoising.
"""

import cv2
import numpy as np
import os
import argparse
from faldoi_deep import run_process
import subprocess
import tempfile
from auxiliar_faldoi_functions import cut_deep_list, delete_outliers

def read_flo_file(filename):
    """Read optical flow from .flo file."""
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise Exception('Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        flow = data.reshape((h, w, 2))
    return flow

def warp_frame(frame, flow):
    """Warp frame according to optical flow."""
    h, w = frame.shape[:2]
    flow_map = np.zeros((h, w, 2), np.float32)
    flow_map[:,:,0] = np.arange(w)
    flow_map[:,:,1] = np.arange(h)[:,np.newaxis]
    flow_map += flow
    return cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)

def temporal_average(frames):
    """Average 5 frames temporally."""
    return np.mean(frames, axis=0).astype(np.uint8)

def process_video(input_video, output_video, work_dir="../Results/video_denoising/"):
    """Process video for denoising using optical flow and temporal averaging."""
    # Create working directory if it doesn't exist
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception("Could not open input video")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Read first 5 frames
    frames_buffer = []
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)
    
    if len(frames_buffer) < 5:
        raise Exception("Video must have at least 5 frames")
    
    # Process frames
    frame_idx = 0
    while True:
        # Save current frames to temporary files for optical flow estimation
        frame_files = []
        for i, frame in enumerate(frames_buffer):
            temp_file = os.path.join(work_dir, f"frame_{i}.png")
            cv2.imwrite(temp_file, frame)
            frame_files.append(temp_file)
        
        # Estimate optical flow between adjacent frames
        warped_frames = [frames_buffer[2]]  # Center frame
        
        # Forward flow (from center to next frames)
        for i in range(2, 4):
            # Create file list for FALDOI
            with open(os.path.join(work_dir, "frame_list.txt"), "w") as f:
                f.write(f"{frame_files[i]}\n{frame_files[i+1]}\n")
            
            # Run FALDOI deep matching
            subprocess.run(["python3", "faldoi_deep.py", 
                          os.path.join(work_dir, "frame_list.txt"),
                          "-res_path", work_dir])
            
            # Read flow and warp frame
            flow = read_flo_file(os.path.join(work_dir, f"frame_{i}_dm_var.flo"))
            warped = warp_frame(frames_buffer[i+1], flow)
            warped_frames.append(warped)
        
        # Backward flow (from center to previous frames)
        for i in range(2, 0, -1):
            # Create file list for FALDOI
            with open(os.path.join(work_dir, "frame_list.txt"), "w") as f:
                f.write(f"{frame_files[i]}\n{frame_files[i-1]}\n")
            
            # Run FALDOI deep matching
            subprocess.run(["python3", "faldoi_deep.py", 
                          os.path.join(work_dir, "frame_list.txt"),
                          "-res_path", work_dir])
            
            # Read flow and warp frame
            flow = read_flo_file(os.path.join(work_dir, f"frame_{i}_dm_var.flo"))
            warped = warp_frame(frames_buffer[i-1], flow)
            warped_frames.insert(0, warped)
        
        # Perform temporal averaging
        denoised_frame = temporal_average(warped_frames)
        
        # Write denoised frame
        out.write(denoised_frame)
        
        # Read next frame
        ret, new_frame = cap.read()
        if not ret:
            break
            
        # Update frame buffer
        frames_buffer.pop(0)
        frames_buffer.append(new_frame)
        frame_idx += 1
        
        print(f"Processed frame {frame_idx}/{total_frames}")
    
    # Clean up
    cap.release()
    out.release()
    
    # Clean temporary files
    for file in os.listdir(work_dir):
        if file.startswith("frame_"):
            os.remove(os.path.join(work_dir, file))

def main():
    parser = argparse.ArgumentParser(description='Video denoising using optical flow')
    parser.add_argument('input_video', help='Path to input video')
    parser.add_argument('output_video', help='Path to output video')
    parser.add_argument('--work_dir', default='../Results/video_denoising/',
                      help='Working directory for temporary files')
    
    args = parser.parse_args()
    process_video(args.input_video, args.output_video, args.work_dir)

if __name__ == "__main__":
    main() 