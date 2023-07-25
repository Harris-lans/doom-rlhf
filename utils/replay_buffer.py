import numpy as np
from utils.segment import Segment

class ReplayBuffer:
    def __init__(self, num_steps, num_envs, observation_space, action_space):
        """
        Initialize the ReplayBuffer.

        Parameters:
            num_steps: int
                The number of steps in each environment trajectory.
            num_envs: int
                The number of parallel environments (usually the batch size).
            observation_space: gym.Space
                The observation space of the environment.
            action_space: gym.Space
                The action space of the environment.
        """
        self.observations = np.zeros((num_steps, num_envs) + observation_space.shape, dtype=observation_space.dtype)
        self.actions = np.zeros((num_steps, num_envs) + action_space.shape, dtype=action_space.dtype)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.int8)
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
    
    def __getitem__(self, index):
        """
        Get the data at the specified index in the ReplayBuffer.

        Parameters:
            index: int
                The index of the data to retrieve.

        Returns:
            tuple
                A tuple containing observations, actions, log probabilities, rewards, values, and done flags at the specified index.
        """
        return self.observations[index], self.actions[index], self.log_probs[index], self.rewards[index], self.values[index], self.dones[index]

    def __setitem__(self, index, value):
        """
        Set the data at the specified index in the ReplayBuffer.

        Parameters:
            index: int
                The index where the data will be set.
            value: tuple
                A tuple containing observations, actions, log probabilities, rewards, values, and done flags to be set at the specified index.
        """
        self.observations[index] = value[0]
        self.actions[index] = value[1]
        self.log_probs[index] = value[2]
        self.rewards[index] = value[3]
        self.values[index] = value[4]
        self.dones[index] = value[5]

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
                A tuple containing observations, actions, log probabilities, rewards, values, and done flags.
        Raises:
            StopIteration
                When the iteration is complete.
        """
        if self.index < self.num_steps:
            result = (self.observations[self.index], self.actions[self.index], self.log_probs[self.index], self.rewards[self.index], self.values[self.index], self.dones[self.index])
            self.index += 1
            return result
        else:
            raise StopIteration
        
    def get_episodic_segments(self, max_episodic_length: int):
        segments = []

        for env in range(self.num_envs):
            current_segment_start_step = 0
            for step in range(self.num_steps):
                num_steps_in_segment = step - current_segment_start_step
                if self.dones[step][env] == 1 or step == self.num_steps - 1 or num_steps_in_segment >= max_episodic_length:
                    segment = Segment(num_steps_in_segment, self.observation_space, self.action_space)
                    segment_step = 0
                    for buffer_step in range(current_segment_start_step, step):
                        segment[segment_step] = (self.observations[buffer_step][env], 
                                                 self.actions[buffer_step][env],
                                                 self.log_probs[buffer_step][env],
                                                 self.rewards[buffer_step][env],
                                                 self.values[buffer_step][env],
                                                 self.dones[buffer_step][env])
                        segment_step += 1
                    segments.append(segment)

                    current_segment_start_step = step + 1

        return segments