import numpy as np

class Memory:
    def __init__(self):
        self.states = []
        self.probabilities = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def num_experiences(self):
        return len(self.dones)

    def generate_batches(self, batch_size):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + batch_size] for i in batch_start]

        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probabilities),\
            np.array(self.values),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def remember(self, state, action, probability, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probability)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probabilities = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
