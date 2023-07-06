from utils.networks import ppo_layer_init
from gym.spaces import Discrete
import torch.nn as nn
import torch.optim as optim
import torch

class DoomRewardPredictor(nn.Module):
    def __init__(self, input_shape, hidden_size=256, use_gpu=True):
        # Creating network
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LSTM(input_size=64 * 11 * 16, hidden_size=hidden_size, num_layers=1, batch_first=True),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Creating optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Choosing the device to run agent on
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
        elif torch.has_mps and use_gpu:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Running the agent on the device
        self.to(self.device)
    
    def train(self, model, dataloader, num_epochs, learning_rate):
      criterion = nn.MSELoss()

      for epoch in range(num_epochs):
          running_loss = 0.0
          for inputs, labels in dataloader:
              self.optimizer.zero_grad()

              outputs = model(inputs)
              loss = criterion(outputs, labels)

              loss.backward()
              self.optimizer.step()

              running_loss += loss.item()

          epoch_loss = running_loss / len(dataloader)
          print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    
    def predict(self):
        x = self.network(x)
        return x.view(x.size(0), -1)
