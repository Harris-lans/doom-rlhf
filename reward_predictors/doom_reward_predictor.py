import torch
from torch import nn
import numpy as np

class DoomRewardPredictor(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_size=256, learning_rate=1e-3, use_gpu=True):
        super(DoomRewardPredictor, self).__init__()

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

        # self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)

        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
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

    def forward(self, observations, actions):
        # Calculating observation and action encodings
        observation_encodings = self.observation_encoder(observations)
        action_encodings = self.action_encoder(actions.unsqueeze(-1).to(torch.float32))
        
        # Using to learn time series data
        lstm_input = torch.cat((observation_encodings, action_encodings), dim=-1)
        # lstm_output, _ = self.lstm(lstm_input)

        reward_predictions = self.reward_predictor(lstm_input)
        return reward_predictions

    def train(self, observations, actions, rewards, batch_size, mini_batch_size=4, num_training_epochs=4):
        # Flatten the data
        flattened_observations = observations.reshape((-1,) + self.observation_shape)
        flattened_actions = actions.reshape((-1,) + tuple() if isinstance(self.action_shape, int) else self.action_shape)
        flattened_rewards = rewards.reshape(-1)

        batch_indices = np.arange(batch_size)

        for epoch in range(num_training_epochs):
            np.random.shuffle(batch_indices)
            epoch_loss = 0
            n = 0

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mini_batch_indices = batch_indices[start:end]
                self.optimizer.zero_grad()
                reward_predictions = self.forward(flattened_observations[mini_batch_indices], flattened_actions.long()[mini_batch_indices])
                loss = self.loss_fn(reward_predictions.squeeze(), flattened_rewards[mini_batch_indices])
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n += 1

            epoch_loss /= n
            print(f'Epoch: {epoch + 1}/{num_training_epochs}, Loss: {epoch_loss:.4f}')

