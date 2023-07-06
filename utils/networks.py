import numpy as np
import torch.nn as nn

def ppo_layer_init(layer, std=np.sqrt(2), bias_constant=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_constant)
    return layer

def build_doom_ppo_agent_networks(observation_space, action_space):
    # Creating networks
    network = nn.Sequential(
        ppo_layer_init(nn.Conv2d(observation_space.shape[0], 32, 8, stride=4)),
        nn.ReLU(),
        ppo_layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        ppo_layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        ppo_layer_init(nn.Linear(64 * 11 * 16, 512)),
        nn.ReLU(),
    )
    actor = ppo_layer_init(nn.Linear(512, action_space.n if isinstance(action_space, Discrete) else action_space), std=0.01)
    critic = ppo_layer_init(nn.Linear(512, 1), std=1)
    pass

def build_reward_predictor_ppo_agent_networks():
    pass