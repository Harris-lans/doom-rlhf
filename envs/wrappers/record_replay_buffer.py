import gymnasium as gym
from envs.doom_env import DoomEnv
from utils.replay_buffer import ReplayBuffer

class RecordReplayBuffer(gym.Wrapper):
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, env_index: int):
        super(RecordReplayBuffer, self).__init__(env)
        self.step = 0

    def step(self, action):
        # Perform the action and get the next state and other information from the environment
        next_observation, reward, done, info = self.env.step(action)

        # Record the current transition in the replay buffer
        # Assuming the replay buffer accepts the following arguments: observation, action, reward, next_observation, done
        self.replay_buffer.record_transition_step(
            self.env.observations,
            action,
            self.env.log_probs,
            reward,
            self.env.values,
            done
        )

        # Return the next state and other information as usual
        return next_observation, reward, done, info