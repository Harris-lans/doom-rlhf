import cv2
from utils.segment import Segment
import os

def generate_video_from_doom_play_segments(segment: Segment, output_path: str, fps: int = 30):
    """
    Generates a video from a segment of Doom gameplay frames and saves it to the specified output path.

    Parameters:
        segment (Segment): The segment containing raw gameplay frames.
        output_path (str): The path to save the generated video.
        fps (int, optional): Frames per second for the output video. Defaults to 30.

    Note:
        This function assumes that the frames in the segment are in RGB format.

    Example:
        generate_video_from_doom_play_segments(segment, "output_video.mp4", fps=30)
    """
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
