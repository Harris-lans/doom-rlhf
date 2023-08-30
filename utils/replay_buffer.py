import numpy as np
from utils.segment import Segment
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, num_steps: int, num_envs: int, raw_observation_space: gym.spaces.Box, processed_observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        """
        Initialize the ReplayBuffer.

        Parameters:
            num_steps: int
                The number of steps in each environment trajectory.
            num_envs: int
                The number of parallel environments (usually the batch size).
            processed_observation_space: gym.Space
                The observation space of the environment.
            action_space: gym.Space
                The action space of the environment.
        """
        self.raw_observations = np.zeros((num_steps, num_envs) + raw_observation_space.shape, dtype=raw_observation_space.dtype)
        self.processed_observations = np.zeros((num_steps, num_envs) + processed_observation_space.shape, dtype=processed_observation_space.dtype)
        self.actions = np.zeros((num_steps, num_envs) + action_space.shape, dtype=action_space.dtype)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.terminations = np.zeros((num_steps, num_envs), dtype=np.int8)
        
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.raw_observation_space = raw_observation_space
        self.processed_observation_space = processed_observation_space
        self.action_space = action_space
    
    def __getitem__(self, index):
        """
        Get the data at the specified index in the ReplayBuffer.

        Parameters:
            index: int
                The index of the data to retrieve.

        Returns:
            tuple
                A tuple containing observations, actions, log probabilities, rewards, values, and termination status at the specified index.
        """
        return self.raw_observations[index], self.processed_observations[index], self.actions[index], self.log_probs[index], self.rewards[index], self.values[index], self.terminations[index]

    def __setitem__(self, index, value):
        """
        Set the data at the specified index in the ReplayBuffer.

        Parameters:
            index: int
                The index where the data will be set.
            value: tuple
                A tuple containing observations, actions, log probabilities, rewards, values, and termination status to be set at the specified index.
        """
        self.raw_observations[index] = value[0]
        self.processed_observations[index] = value[1]
        self.actions[index] = value[2]
        self.log_probs[index] = value[3]
        self.rewards[index] = value[4]
        self.values[index] = value[5]
        self.terminations[index] = value[6]

    def __len__(self):
        """
        Get the number of steps in the ReplayBuffer.

        Returns:
            int
                The number of steps in the ReplayBuffer.
        """
        return self.num_steps

    def __iter__(self):
        """
        Initialize the iterator for the ReplayBuffer.

        Returns:
            ReplayBuffer
                The iterator object.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Get the next data in the ReplayBuffer during iteration.

        Returns:
            tuple
                A tuple containing observations, actions, log probabilities, rewards, values, and termination status.
        Raises:
            StopIteration
                When the iteration is complete.
        """
        if self.index < self.num_steps:
            result = (self.raw_observations[self.index], self.processed_observations[self.index], self.actions[self.index], self.log_probs[self.index], self.rewards[self.index], self.values[self.index], self.terminations[self.index])
            self.index += 1
            return result
        else:
            raise StopIteration
    
    def get_segments(self, segment_length: int):
        segments = []
        
        assert self.num_steps % segment_length == 0, "The segment length must be a divisible of the number of steps in the segment"

        # Looping through all the steps, one environment at a time
        for env in range(self.num_envs):
            # Creating segments of the provided size
            for step in range(0, self.num_steps, segment_length):
                current_segment_start_step = step
                segment = Segment(segment_length, 
                                    self.raw_observation_space, 
                                    self.processed_observation_space, 
                                    self.action_space)
                segment_step = 0
                for buffer_step in range(current_segment_start_step, current_segment_start_step + segment_length):
                    segment[segment_step] = (self.raw_observations[buffer_step][env], 
                                             self.processed_observations[buffer_step][env],
                                             self.actions[buffer_step][env],
                                             self.log_probs[buffer_step][env],
                                             self.rewards[buffer_step][env],
                                             self.values[buffer_step][env],
                                             self.terminations[buffer_step][env])
                    segment_step += 1
                segments.append(segment)

        return segments