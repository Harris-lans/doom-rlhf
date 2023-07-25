from reward_predictors.base_human_preference_reward_predictor import BaseHumanPreferenceRewardPredictor
from utils.networks import ppo_layer_init
from gymnasium.spaces import Discrete, Space
import torch.nn as nn

class DoomHumanPreferenceRewardPredictor(BaseHumanPreferenceRewardPredictor):
    """Custom PPO agent specifically designed for the Doom environment."""
    
    def __init__(self, observation_space: Space, action_space: Space, hidden_size=512, learning_rate=0.00025, use_gpu=True):
        """
        Initialize the DoomPpoAgent.

        Parameters:
            observation_space (gymnasium.Space): The observation space of the environment.
            action_space (gymnasium.Space): The action space of the environment.
            models_path (str): The path to the directory where models will be saved.
            learning_rate (float): The learning rate for the optimizer.
            use_gpu (bool): Whether to use GPU for training.
        """

        observation_encoder = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 16, hidden_size),
            nn.ReLU(),
        )

        action_encoder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU()
        )

        reward_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1 if isinstance(action_space, Discrete) else action_space.shape)
        )

        super().__init__(observation_encoder, action_encoder, reward_predictor, observation_space.shape, action_space.shape, learning_rate, use_gpu)
