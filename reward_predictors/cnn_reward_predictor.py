import torch
from torch import nn
import numpy as np
import os


class CnnRewardPredictor(nn.Module):
    """A convolutional neural network model for reward prediction.

    Args:
        observation_shape (tuple): The shape of the input observations.
        action_shape (int or tuple): The shape of the input actions.
        model_path (str, optional): Path to load pre-trained models from. Defaults to None.
        hidden_size (int, optional): The size of the hidden layers. Defaults to 256.
        learning_rate (float, optional): The learning rate for optimization. Defaults to 1e-3.
        use_gpu (bool, optional): Whether to use GPU if available. Defaults to True.
    """

    def __init__(self, observation_shape, action_shape, model_path=None, hidden_size=256, learning_rate=1e-3, use_gpu=True):
        super(CnnRewardPredictor, self).__init__()

        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.observation_encoder = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 16, hidden_size),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        if model_path is not None:
            self.load_models(model_path)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.loss_fn = nn.MSELoss()

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

    def load_models(self, path='./models'):
        """Load pre-trained models from a file.

        Args:
            path (str, optional): Directory containing the models. Defaults to './models'.
        """
        # Checking if the models exist in the provided path
        assert os.path.exists(path), "Given path is invalid."
        assert os.path.isfile(f"{path}/reward_predictor.pth"), "The given path does not contain the model."

        # Loading models
        print("Loading models...")
        model_dict = torch.load(f"{path}/reward_predictor.pth")
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

    def train(self, observations, actions, rewards, batch_size, mini_batch_size=4, num_training_epochs=4):
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
        # Flatten the data
        flattened_observations = observations.reshape((-1,) + self.observation_shape)
        flattened_actions = actions.reshape((-1,) + tuple() if isinstance(self.action_shape, int) else self.action_shape)
        flattened_rewards = rewards.reshape(-1)

        batch_indices = np.arange(batch_size)

        total_loss = 0.0
        total_samples = 0

        for epoch in range(num_training_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mini_batch_indices = batch_indices[start:end]
                self.optimizer.zero_grad()
                reward_predictions = self.forward(
                    flattened_observations[mini_batch_indices],
                    flattened_actions.long()[mini_batch_indices]
                )
                loss = self.loss_fn(reward_predictions.squeeze(), flattened_rewards[mini_batch_indices])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_samples += 1

            # self.scheduler.step()

        avg_loss = total_loss / total_samples

        # Training statistics
        training_stats = {
            'avg_loss': avg_loss
        }

        return training_stats
