import cv2
from utils.segment import Segment

def generate_video_from_grayscale_doom_play_segments(segment: Segment, output_path: str, fps: int = 30):
    print("Generating video from segment...")
    num_steps, num_channels, frame_height, frame_width = segment.observations.shape
    print(f"segment.observations.shape={segment.observations.shape}")
    frames = segment.observations.reshape(num_steps * num_channels, frame_height, frame_width)
    print(f"frames.shape={frames.shape}")

    # Define the codec and create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec for MP4 file format
    output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), 0)

    i = 0
    for frame in frames:
        print(f"Writing frame {i}...")
        # Convert the grayscale frame to BGR format
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Write the BGR frame to the output video
        output.write(bgr_frame)
        print(f"Successfully wrote frame {i}!")
        i += 1

    # Release video sources and writer
    output.release()
    print("Generated video from segment!")