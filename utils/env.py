import gymnasium as gym
from envs.vizdoom_env import VizdoomEnv
from utils.replay_buffer import ReplayBuffer
from envs.wrappers.record_replay_buffer import RecordReplayBuffer

def make_vizdoom_env(level: str, env_id=0, unprocessed_frames_replay_buffer: ReplayBuffer = None, render_mode=None, frame_stack_size=4):
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
        # Recording transitions with unprocessed transitions
        if unprocessed_frames_replay_buffer is not None:
            env = RecordReplayBuffer(env, env_id, unprocessed_frames_replay_buffer)

        # Observation Pre-Processing Wrappers
        env = gym.wrappers.ResizeObservation(env, (120, 160))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, frame_stack_size)

        return env

    return thunk
