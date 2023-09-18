import numpy as np
import gymnasium as gym

class Segment:
    def __init__(self, num_steps: int, raw_observation_space: gym.spaces.Box, processed_observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        """
        Initializes a Segment object.

        Parameters:
            num_steps (int): The number of steps in the segment.
            raw_observation_space (gym.spaces.Box): The observation space of the environment.
            processed_observation_space (gym.spaces.Box): The processed observation space of the environment.
            action_space (gym.spaces.Discrete): The action space of the environment.
        """
        self.raw_observations = np.zeros((num_steps,) + raw_observation_space.shape, dtype=raw_observation_space.dtype)
        self.processed_observations = np.zeros((num_steps,) + processed_observation_space.shape, dtype=processed_observation_space.dtype)
        self.actions = np.zeros((num_steps,) + action_space.shape, dtype=action_space.dtype)
        self.log_probs = np.zeros((num_steps,), dtype=np.float32)
        self.rewards = np.zeros((num_steps,), dtype=np.float32)
        self.values = np.zeros((num_steps,), dtype=np.float32)
        self.terminations = np.zeros((num_steps,), dtype=np.uint8)
        self.num_steps = num_steps

        self.raw_observation_space = raw_observation_space
        self.processed_observation_space = processed_observation_space
        self.action_space = action_space

    def __getitem__(self, index):
        """
        Returns the data at the specified index in the segment.

        Parameters:
            index (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing observation, action, log probability, reward, value, and termination status at the specified index.
        """
        return self.raw_observations[index], self.processed_observations[index], self.actions[index], self.log_probs[index], self.rewards[index], self.values[index], self.terminations[index]

    def __setitem__(self, index, value):
        """
        Set the data at the specified index in the segment.

        Parameters:
            index (int): The index where the data will be set.
            value (tuple): A tuple containing observations, actions, log probabilities, rewards, values, and termination status to be set at the specified index.
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
        Get the number of steps in the segment.

        Returns:
            int: The number of steps in the segment.
        """
        return self.num_steps

    def __iter__(self):
        """
        Initialize the iterator for the segment.

        Returns:
            Segment: The iterator object.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Get the next data in the segment during iteration.

        Returns:
            tuple: A tuple containing observations, actions, log probabilities, rewards, values, and termination status.
        
        Raises:
            StopIteration: When the iteration is complete.
        """
        if self.index < self.num_steps:
            result = (self.raw_observations[self.index], self.processed_observations[self.index], self.actions[self.index], self.log_probs[self.index], self.rewards[self.index], self.values[self.index], self.terminations[self.index])
            self.index += 1
            return result
        else:
            raise StopIteration
