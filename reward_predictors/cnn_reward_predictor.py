import torch
from torch import nn
import numpy as np
import os

class CnnRewardPredictor(nn.Module):
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
        # Calculating observation and action encodings
        observation_encodings = self.observation_encoder(observations)
        action_encodings = self.action_encoder(actions.unsqueeze(-1).to(torch.float32))
        
        # Predicting rewards
        reward_predictions_input = torch.cat((observation_encodings, action_encodings), dim=-1)
        reward_predictions = self.reward_predictor(reward_predictions_input)

        return reward_predictions

    def train(self, observations, actions, rewards, batch_size, mini_batch_size=4, num_training_epochs=4):
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
                reward_predictions = self.forward(flattened_observations[mini_batch_indices], flattened_actions.long()[mini_batch_indices])
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
    
    def train_cv(self, observations, actions, rewards, batch_size, mini_batch_size=4, num_training_epochs=4, cv=5):
        # Flatten the data
        flattened_observations = observations.reshape((-1,) + self.observation_shape)
        flattened_actions = actions.reshape((-1,) + tuple() if isinstance(self.action_shape, int) else self.action_shape)
        flattened_rewards = rewards.reshape(-1)

        fold_size = batch_size // cv

        # Shuffle the data randomly
        indices = np.random.permutation(batch_size)
        observations = flattened_observations[indices]
        actions = flattened_actions[indices]
        rewards = flattened_rewards[indices]

        fold_metrics = []  # List to store metrics for each fold

        for fold in range(cv):
            # Split data into training and validation sets
            validation_start = fold * fold_size
            validation_end = validation_start + fold_size
            val_observations = observations[validation_start:validation_end]
            val_actions = actions[validation_start:validation_end]
            val_rewards = rewards[validation_start:validation_end]

            train_observations = np.concatenate(
                [observations[:validation_start], observations[validation_end:]], axis=0
            )
            train_actions = np.concatenate(
                [actions[:validation_start], actions[validation_end:]], axis=0
            )
            train_rewards = np.concatenate(
                [rewards[:validation_start], rewards[validation_end:]], axis=0
            )

            # Train the model on the training set
            training_loss = 0.0
            training_samples = 0

            for epoch in range(num_training_epochs):
                batch_indices = np.arange(train_observations.shape[0])
                np.random.shuffle(batch_indices)

                for start in range(0, train_observations.shape[0], mini_batch_size):
                    end = start + mini_batch_size
                    mini_batch_indices = batch_indices[start:end]
                    self.optimizer.zero_grad()
                    reward_predictions = self.forward(train_observations[mini_batch_indices], train_actions.long()[mini_batch_indices])
                    loss = self.loss_fn(reward_predictions.squeeze(), train_rewards[mini_batch_indices])
                    loss.backward()
                    self.optimizer.step()
                    training_loss += loss.item()
                    training_samples += 1

                # self.scheduler.step()

            avg_training_loss = training_loss / training_samples

            # Evaluate the model on the validation set
            validation_predictions = self.forward(val_observations, val_actions)
            validation_loss = self.loss_fn(validation_predictions.squeeze(), val_rewards)
            fold_metrics.append(validation_loss.item())

        # Calculate aggregate performance metrics
        mean_loss = np.mean(fold_metrics)
        std_loss = np.std(fold_metrics)

        print("Cross-validation results:")
        print(f"Mean Loss: {mean_loss:.4f}")
        print(f"Standard Deviation Loss: {std_loss:.4f}")