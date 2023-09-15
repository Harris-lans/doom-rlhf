import gymnasium as gym
from envs.vizdoom_env import VizdoomEnv
from envs.wrappers.record_observations import RecordObservations

def make_vizdoom_env(level: str, render_mode=None, frame_stack_size=4):
    """Factory function to create a Doom environment with specified configurations.

    Args:
        level (str): Path to the Doom level configuration file.
        render_mode (str, optional): Rendering mode for the environment. Defaults to None.
        frame_stack_size (int, optional): Number of frames to stack as input. Defaults to 4.

    Returns:
        function: A thunk that creates the Doom environment.
    """
    def thunk():
        env = VizdoomEnv(level, render_mode=render_mode)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = RecordObservations(env, 'raw_observations')

        # Observation Pre-Processing Wrappers
        env = gym.wrappers.ResizeObservation(env, (120, 160))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, frame_stack_size)

        return env

    return thunk
