import torch
from torch import nn
import numpy as np
import os
from losses.preference_loss import PreferenceLoss
from utils.segment import Segment
import math
from typing import Tuple

class BaseHumanPreferenceRewardPredictor(nn.Module):
    """A convolutional neural network model for reward prediction.

    Args:
        observation_shape (tuple): The shape of the input observations.
        action_shape (int or tuple): The shape of the input actions.
        model_path (str, optional): Path to load pre-trained models from. Defaults to None.
        hidden_size (int, optional): The size of the hidden layers. Defaults to 256.
        learning_rate (float, optional): The learning rate for optimization. Defaults to 1e-3.
        use_gpu (bool, optional): Whether to use GPU if available. Defaults to True.
    """

    def __init__(self, observation_encoder: nn.Sequential, action_encoder: nn.Sequential, reward_predictor: nn.Sequential, observation_shape: Tuple[int, ...], action_shape: Tuple[int, ...], learning_rate=1e-3, use_gpu=True):
        super(BaseHumanPreferenceRewardPredictor, self).__init__()

        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder
        self.reward_predictor = reward_predictor

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.loss_fn = PreferenceLoss()

        # Choosing the device to run agent on
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
        elif torch.has_mps and use_gpu:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Running the agent on the device
        self.to(self.device)

    def save_models(self, path='./models'):
        """Save the model's weights to a file.

        Args:
            path (str, optional): Directory to save the models. Defaults to './models'.
        """
        print("Saving models...")

        # Creating directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created!")
        else:
            print(f"Directory '{path}' already exists!")

        # Saving network states
        torch.save({
            'observation_encoder': self.observation_encoder.state_dict(),
            'action_encoder': self.action_encoder.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict()
        }, f"{path}/reward_predictor.pth")

        print("Successfully saved models!")

    def load_models(self, path):
        """Load pre-trained models from a file.

        Args:
            path (str): Directory containing the models.
        """
        # Checking if the models exist in the provided path
        assert os.path.exists(path), "Given path is invalid."
        assert os.path.isfile(f"{path}/reward_predictor.pth"), "The given path does not contain the model."

        # Loading models
        print("Loading models...")
        model_dict = torch.load(f"{path}/reward_predictor.pth", map_location=self.device)
        print("Successfully loaded models!")

        # Updating networks with loaded models
        print("Updating networks with weights from loaded models...")
        self.observation_encoder.load_state_dict(model_dict['observation_encoder'])
        self.action_encoder.load_state_dict(model_dict['action_encoder'])
        self.reward_predictor.load_state_dict(model_dict['reward_predictor'])
        print("Successfully updated networks!")

    def forward(self, observations, actions):
        """Forward pass of the reward predictor.

        Args:
            observations (np.Array): Input observations.
            actions (np.Array): Input actions.

        Returns:
            torch.Tensor: Predicted rewards.
        """
        observations = torch.tensor(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)

        with torch.no_grad():
            # Calculating observation and action encodings
            observation_encodings = self.observation_encoder(observations / 255.0)
            action_encodings = self.action_encoder(actions.unsqueeze(-1).to(torch.float32))
            # action_encodings = self.action_encoder(actions.to(torch.float32))

            # Predicting rewards
            reward_predictions_input = torch.cat((observation_encodings, action_encodings), dim=-1)
            reward_predictions = self.reward_predictor(reward_predictions_input)

        return reward_predictions.squeeze().cpu().numpy()
    
    def _training_forward(self, observations: torch.Tensor, actions: torch.Tensor):
        """Forward pass of the reward predictor.

        Args:
            observations (torch.Tensor): Input observations.
            actions (torch.Tensor): Input actions.

        Returns:
            torch.Tensor: Predicted rewards.
        """
        # Calculating observation and action encodings
        observation_encodings = self.observation_encoder(observations)
        action_encodings = self.action_encoder(actions.unsqueeze(-1).to(torch.float32))

        # Predicting rewards
        reward_predictions_input = torch.cat((observation_encodings, action_encodings), dim=-1)
        reward_predictions = self.reward_predictor(reward_predictions_input)

        return reward_predictions

    def train(self, segment_1: Segment, segment_2: Segment, preference: float, epochs: int = 4):
        """Train the reward predictor model.

        Args:
            observations (torch.Tensor): Input observations.
            actions (torch.Tensor): Input actions.
            rewards (torch.Tensor): Corresponding rewards.
            batch_size (int): Batch size.
            mini_batch_size (int, optional): Mini-batch size. Defaults to 4.
            num_training_epochs (int, optional): Number of training epochs. Defaults to 4.

        Returns:
            dict: Training statistics.
        """
        assert preference in (0, 0.5, 1), "Invalid preference value"

        total_loss = 0

        for i in range(epochs):
            # Resetting gradients
            self.optimizer.zero_grad()

            # Calculating sum of latent rewards for both segments
            segment_1_rewards = self._training_forward(torch.tensor(segment_1.observations).to(self.device),
                                                    torch.tensor(segment_1.actions).to(self.device))
            segment_2_rewards = self._training_forward(torch.tensor(segment_2.observations).to(self.device),
                                                    torch.tensor(segment_2.actions).to(self.device))
            segment_1_latent_rewards_sum = torch.sum(segment_1_rewards).item()
            segment_2_latent_rewards_sum = torch.sum(segment_2_rewards).item()

            # Calculating probability of preference of the segments
            prob_prefer_segment_1 = math.exp(segment_1_latent_rewards_sum) / math.exp(segment_1_latent_rewards_sum) + math.exp(segment_2_latent_rewards_sum)
            prob_prefer_segment_2 = math.exp(segment_2_latent_rewards_sum) / math.exp(segment_1_latent_rewards_sum) + math.exp(segment_2_latent_rewards_sum)

            # Calculating loss
            loss = self.loss_fn(prob_prefer_segment_1, prob_prefer_segment_2, preference)

            # Updating gradients
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Training statistics
        training_stats = {
            'loss': total_loss / epochs
        }

        return training_stats
