import gymnasium as gym
from envs.doom_env import DoomEnv
from envs.vizdoom_env import VizdoomEnv
from utils.time import current_timestamp_ms

def make_vizdoom_env(level: str, render_mode=None, frame_stack_size=4, record_episodes=False, recording_save_path=f"videos/{current_timestamp_ms()}", recording_file_prefix=f"vizdoom"):
    """Factory function to create a Doom environment with specified configurations.

    Args:
        level_config_path (str): Path to the Doom level configuration file.
        frame_stack_size (int, optional): Number of frames to stack as input. Defaults to 4.
        record_episodes (bool, optional): Whether to record episodes. Defaults to False.
        render (bool, optional): Whether to render the environment. Defaults to True.

    Returns:
        function: A thunk that creates the Doom environment.
    """
    def thunk():
        env = VizdoomEnv(level, render_mode=render_mode)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record_episodes:
            env = gym.wrappers.RecordVideo(env, recording_save_path, name_prefix=recording_file_prefix, disable_logger=False)

        # Observation wrappers
        env = gym.wrappers.ResizeObservation(env, (120, 160))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, frame_stack_size)

        return env

    return thunk
