import gymnasium as gym
from utils.replay_buffer import ReplayBuffer

class RecordReplayBuffer(gym.Wrapper):
    def __init__(self, env: gym.Env, env_index: int, replay_buffer: ReplayBuffer):
        super(RecordReplayBuffer, self).__init__(env)
        self.replay_buffer = replay_buffer
        self.env_index = env_index

        self.step
        self.observation = None
        self.done = 0

    def reset(self):
        # Call the reset method of the wrapped environment
        observation, info = self.env.reset()

        # Updating variables
        self.observation = observation
        self.done = 0
        self.step = 0

        return observation, info

    def step(self, action):
        # Perform the action and get the next state and other information from the environment
        next_observation, reward, next_done, truncated, info = self.env.step(action)

        # Record the current transition in the replay buffer
        self.replay_buffer.observations[self.step % self.replay_buffer.num_steps][self.env_index] = self.observation
        self.replay_buffer.actions[self.step % self.replay_buffer.num_steps][self.env_index] = action
        self.replay_buffer.rewards[self.step % self.replay_buffer.num_steps][self.env_index] = reward
        self.replay_buffer.dones[self.step % self.replay_buffer.num_steps][self.env_index] = self.done

        self.step += 1

        self.observation = next_observation
        self.done = next_done

        # Return the next state and other information as usual
        return next_observation, reward, next_done, truncated, info