from agents.ppo_agent import PpoAgent
from utils.networks import ppo_layer_init
from gym.spaces import Discrete
import torch.nn as nn

class DoomPpoAgent(PpoAgent):
    """Custom PPO agent specifically designed for the Doom environment."""
    
    def __init__(self, observation_space, action_space, models_path=None, learning_rate=0.00025, use_gpu=True):
        """
        Initialize the DoomPpoAgent.

        Parameters:
            observation_space (gym.Space): The observation space of the environment.
            action_space (gym.Space): The action space of the environment.
            models_path (str): The path to the directory where models will be saved.
            learning_rate (float): The learning rate for the optimizer.
            use_gpu (bool): Whether to use GPU for training.
        """
        base_network = nn.Sequential(
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
        actor_network = ppo_layer_init(nn.Linear(512, action_space.n if isinstance(action_space, Discrete) else action_space), std=0.01)
        critic_network = ppo_layer_init(nn.Linear(512, 1), std=1)

        super().__init__(base_network, actor_network, critic_network, observation_space.shape, action_space.shape, models_path, learning_rate, use_gpu)
