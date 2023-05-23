import gym
from envs.doom_env import DoomEnv
from utils.time import current_timestamp_ms

def make_doom_env(level_config_path, frame_stack_size=4, record_episodes=False, render=True):
  def thunk():
    env = DoomEnv(level_config_path, render);
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if record_episodes:
      env = gym.wrappers.RecordVideo(env, f"videos/[{level_config_path}]_{current_timestamp_ms()}")
    env = gym.wrappers.FrameStack(env, frame_stack_size)

    return env
  
  return thunk 