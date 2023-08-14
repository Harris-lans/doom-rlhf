import torch
from torch import nn
import numpy as np
import os
from losses.preference_loss import PreferenceLoss
from utils.segment import Segment
from utils.running_stat import RunningStat
from typing import Tuple

FLOAT_EPSILON = 1e-8

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

    def __init__(self, observation_encoder_network: nn.Sequential, reward_predictor_network: nn.Sequential, observation_shape: Tuple[int, ...], learning_rate=1e-3, use_gpu=True):
        super(BaseHumanPreferenceRewardPredictor, self).__init__()

        self.observation_shape = observation_shape

        self.observation_encoder_network = observation_encoder_network
        self.reward_predictor_network = reward_predictor_network

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
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

        self.running_stat = RunningStat(self.device)

    def save_models(self, path='./models'):
        """Save the model's weights to a file.

        Args:
            path (str, optional): Directory to save the models. Defaults to './models'.
        """
        # Creating directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created!")
        else:
            print(f"Directory '{path}' already exists!")
        
        # Saving network states
        torch.save({
            'observation_encoder_network': self.observation_encoder_network.state_dict(),
            'reward_predictor_network': self.reward_predictor_network.state_dict(),
        }, f"{path}/model.pth")

    def load_models(self, path):
        """Load pre-trained models from a file.

        Args:
            path (str): Directory containing the models.
        """
        # Checking if the model exists in the provided path
        assert os.path.exists(path), "Given path is invalid."
        assert os.path.isfile(f"{path}/model.pth"), "The given path does not contain the model."

        # Loading models
        model_dict = torch.load(f"{path}/model.pth", map_location=self.device)

        # Updating networks with loaded models
        self.observation_encoder_network.load_state_dict(model_dict['observation_encoder_network'])
        self.reward_predictor_network.load_state_dict(model_dict['reward_predictor_network'])

    def _normalize_rewards(self, rewards: torch.Tensor):
        # Normalizing rewards
        rewards -= self.running_stat.mean
        rewards /= (self.running_stat.std + torch.tensor(FLOAT_EPSILON).to(self.device))
        rewards *= torch.tensor(0.05).to(self.device)

        return rewards

    def forward(self, observations: np.ndarray):
        """Forward pass of the reward predictor.

        Args:
            observations (np.Array): Input observations.
            actions (np.Array): Input actions.

        Returns:
            torch.Tensor: Predicted rewards.
        """
        observations = torch.tensor(observations).to(self.device)

        with torch.no_grad():
            # Predicting rewards
            observation_encodings = self.observation_encoder_network(observations / 255.0)
            reward_predictions = self.reward_predictor_network(observation_encodings)
            reward_predictions = reward_predictions.squeeze()

            # Pushing calculated reward prediction to running stat
            for reward_prediction in reward_predictions:
                self.running_stat.push(reward_prediction)

            # Normalizing rewards
            reward_predictions = self._normalize_rewards(reward_predictions)

        return reward_predictions.cpu().numpy()
    
    def _training_forward(self, observations: torch.Tensor):
        """Forward pass of the reward predictor.

        Args:
            observations (torch.Tensor): Input observations.
            actions (torch.Tensor): Input actions.

        Returns:
            torch.Tensor: Predicted rewards.
        """
        # Predicting rewards
        observation_encodings = self.observation_encoder_network(observations / 255.0)
        reward_predictions = self.reward_predictor_network(observation_encodings)
        reward_predictions = reward_predictions.squeeze()

        return reward_predictions

    def train(self, segment_1: Segment, segment_2: Segment, preference: float, epochs: int = 1):
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
            # Calculating sum of latent rewards for both segments
            segment_1_rewards = self._training_forward(torch.tensor(segment_1.processed_observations).to(self.device))
            segment_2_rewards = self._training_forward(torch.tensor(segment_2.processed_observations).to(self.device))
            segment_1_latent_rewards_sum = torch.sum(segment_1_rewards)
            segment_2_latent_rewards_sum = torch.sum(segment_2_rewards)

            # Adding a small epsilon value to prevent NaN values during the training process
            segment_1_latent_rewards_sum += torch.tensor(FLOAT_EPSILON).to(self.device)
            segment_2_latent_rewards_sum += torch.tensor(FLOAT_EPSILON).to(self.device)

            # Calculating probability of preference of the segments
            prob_prefer_segment_1 = torch.exp(segment_1_latent_rewards_sum) / (torch.exp(segment_1_latent_rewards_sum) + torch.exp(segment_2_latent_rewards_sum))
            prob_prefer_segment_2 = torch.exp(segment_2_latent_rewards_sum) / (torch.exp(segment_1_latent_rewards_sum) + torch.exp(segment_2_latent_rewards_sum))

            if preference == 0:
                mu1 = 1
                mu2 = 0
            elif preference == 0.5:
                mu1 = 0.5
                mu2 = 0.5
            else:
                mu1 = 0
                mu2 = 1

            mu1 = torch.tensor(mu1).to(self.device)
            mu2 = torch.tensor(mu2).to(self.device)

            # Calculating loss
            loss = self.loss_fn(prob_prefer_segment_1, prob_prefer_segment_2, mu1, mu2)
            # Resetting gradients
            self.optimizer.zero_grad()
            # Updating gradients based on loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Training statistics
        training_stats = {
            'loss': total_loss / epochs
        }

        return training_stats
