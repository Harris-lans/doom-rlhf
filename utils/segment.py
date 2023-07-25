import numpy as np

class Segment:
    def __init__(self, num_steps: int, observation_space, action_space):
        """
        Initializes a Segment object.

        Parameters:
            num_steps: int
                The number of steps in the segment.
            observation_space: gym.Space
                The observation space of the environment.
            action_space: gym.Space
                The action space of the environment.
        """
        self.observations = np.zeros((num_steps,) + observation_space.shape, dtype=observation_space.dtype)
        self.actions = np.zeros((num_steps,) + action_space.shape, dtype=action_space.dtype)
        self.log_probs = np.zeros((num_steps,), dtype=np.float32)
        self.rewards = np.zeros((num_steps,), dtype=np.float32)
        self.values = np.zeros((num_steps,), dtype=np.float32)
        self.dones = np.zeros((num_steps,), dtype=np.uint8)
        self.num_steps = num_steps

    def __getitem__(self, index):
        """
        Returns the data at the specified index in the segment.

        Parameters:
            index: int
                The index of the data to retrieve.

        Returns:
            tuple
                A tuple containing observation, action, log probability, reward, value, and done status at the specified index.
        """
        return self.observations[index], self.actions[index], self.log_probs[index], self.rewards[index], self.values[index], self.dones[index]

    def __setitem__(self, index, value):
        """
        Sets the data at the specified index in the segment.

        Parameters:
            index: int
                The index where the data will be set.
            value: tuple
                A tuple containing observation, action, log probability, reward, value, and done status to be set at the specified index.
        """
        self.observations[index] = value[0]
        self.actions[index] = value[1]
        self.log_probs[index] = value[2]
        self.rewards[index] = value[3]
        self.values[index] = value[4]
        self.dones[index] = value[5]

    def __len__(self):
        """
        Returns the number of steps in the segment.

        Returns:
            int
                The number of steps in the segment.
        """
        return self.num_steps

    def __iter__(self):
        """
        Initializes the iterator for the segment.

        Returns:
            Segment
                The iterator object.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Returns the next data in the segment during iteration.

        Returns:
            tuple
                A tuple containing observation, action, log probability, reward, value, and done status.
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
