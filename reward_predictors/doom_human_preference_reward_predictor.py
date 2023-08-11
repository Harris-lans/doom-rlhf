from reward_predictors.base_human_preference_reward_predictor import BaseHumanPreferenceRewardPredictor
from gymnasium.spaces import Space
import torch.nn as nn
import torch

class DoomHumanPreferenceRewardPredictor(BaseHumanPreferenceRewardPredictor):
    """Custom PPO agent specifically designed for the Doom environment."""
    
    def __init__(self, observation_space: Space, hidden_size=64, learning_rate=0.00025, drop_out=0.5, use_gpu=True):
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
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(drop_out),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(drop_out),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(drop_out),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(drop_out)
        )

        # Calculating the output size of observation encoder network
        with torch.no_grad():
            dummy_input = torch.randn(1, *observation_space.shape)
            conv_output = observation_encoder(dummy_input)
            conv_output_size = torch.prod(torch.tensor(conv_output.size()[1:]))

        reward_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, 1)
        )


        super().__init__(observation_encoder, reward_predictor, observation_space.shape, learning_rate, use_gpu)
