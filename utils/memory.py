import numpy as np
import torch


class Memory:
    """Replay memory for storing environment transitions.

    Args:
        device (torch.device): The device to store the memory tensors.
        num_steps (int): The number of steps to store in memory.
        num_envs (int): The number of parallel environments.
        observation_space_shape (tuple): The shape of the observation space.
        action_space_shape (tuple): The shape of the action space.
    """

    def __init__(self, device, num_steps, num_envs, observation_space_shape, action_space_shape):
        self.observations = torch.zeros((num_steps, num_envs) + observation_space_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + action_space_shape).to(device)
        self.log_probs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)

    def remember(self, step, observation, action, log_prob, reward, value, done):
        """Remember a transition in memory.

        Args:
            step (int): The step index in memory.
            observation (torch.Tensor): The observation.
            action (torch.Tensor): The action.
            log_prob (torch.Tensor): The log probability of the action.
            reward (torch.Tensor): The reward.
            value (torch.Tensor): The estimated value of the state.
            done (torch.Tensor): The done flag indicating if the episode terminated.
        """
        self.observations[step] = observation
        self.actions[step] = action
        self.log_probs[step] = log_prob
        self.rewards[step] = reward
        self.values[step] = value
        self.dones[step] = done
