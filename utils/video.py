import cv2
from utils.segment import Segment
import os
import numpy as np

def generate_video_from_grayscale_doom_play_segments(segment: Segment, output_path: str, fps: int = 30):
    output_path = os.path.abspath(output_path)
    num_steps, num_channels, frame_height, frame_width = segment.processed_observations.shape

    frames = np.zeros((num_steps, frame_height, frame_width), dtype=segment.processed_observations.dtype)
    for i in range(num_steps):
        frames[i] = segment.processed_observations[i][0]

    # Define the codec and create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec for MP4 file format
    output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), 0)

    i = 0
    for frame in frames:
        # Convert the grayscale frame to BGR format
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Write the BGR frame to the output video
        output.write(bgr_frame)
        i += 1

    # Release video sources and writer
    output.release()

def generate_video_from_doom_play_segments(segment: Segment, output_path: str, fps: int = 30):
    output_path = os.path.abspath(output_path)
    num_steps, frame_height, frame_width, num_channels = segment.raw_observations.shape

    frames = segment.raw_observations

    # Define the codec and create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec for MP4 file format
    output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), 0)

    i = 0
    for frame in frames:
        # Convert the RGB frame to BGR format
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write the BGR frame to the output video
        output.write(bgr_frame)
        i += 1

    # Release video sources and writer
    output.release()