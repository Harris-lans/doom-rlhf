import numpy as np

class ReplayBuffer:
    """Replay buffer for storing environment transitions.

    Args:
        device (torch.device): The device to store the memory tensors.
        num_steps (int): The number of steps to store in memory.
        num_envs (int): The number of parallel environments.
        observation_space_shape (tuple): The shape of the observation space.
        action_space_shape (tuple): The shape of the action space.
    """

    def __init__(self, num_steps, num_envs, observation_space, action_space):
        self.observations = np.zeros((num_steps, num_envs) + observation_space.shape, dtype=observation_space.dtype)
        self.actions = np.zeros((num_steps, num_envs) + action_space.shape, dtype=action_space.dtype)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.int8)
        self.num_steps = num_steps
        self.num_envs = num_envs

    def record_transition_step(self, step, observations, actions, log_probs, rewards, values, dones):
        """Add experience to buffer.

        Args:
            observation (array): The observation.
            action (integer): The action.
            log_prob (float): The log probability of the action.
            reward (float): The reward.
            value (float): The estimated value of the state.
            done (integer): The done flag indicating if the episode terminated.
        """
        self.observations[step] = observations
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.rewards[step] = rewards
        self.values[step] = values
        self.dones[step] = dones

