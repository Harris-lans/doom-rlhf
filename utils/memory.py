import numpy as np
import torch

class Memory:
    def __init__(self, device, num_steps, num_envs, observation_space_shape, action_space_shape):
        self.observations = torch.zeros((num_steps, num_envs) + observation_space_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + action_space_shape).to(device)
        self.log_probs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)

    def remember(self, step, observation, action, log_prob, reward, value, done):
        self.observations[step] = observation
        self.actions[step] = action
        self.log_probs[step] = log_prob
        self.rewards[step] = reward
        self.values[step] = value
        self.dones[step] = done
