import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils.time import *
from utils.replay_buffer import ReplayBuffer
import os

class TrainingStats():
    def __init__(
            self,
            learning_rate,
            value_loss,
            policy_loss,
            entropy_loss,
            old_approx_kl,
            approx_kl,
            clip_fraction,
            explained_variance
        ):
        """
        Store the training statistics for PPO training.

        Parameters:
            learning_rate (float): The learning rate used during training.
            value_loss (float): The value loss.
            policy_loss (float): The policy loss.
            entropy_loss (float): The entropy loss.
            old_approx_kl (float): The old approximation KL divergence.
            approx_kl (float): The current approximation KL divergence.
            clip_fraction (float): The fraction of clipped values.
            explained_variance (float): The explained variance of the value predictions.
        """
        self.learning_rate = learning_rate 
        self.value_loss = value_loss
        self.policy_loss = policy_loss
        self.entropy_loss = entropy_loss
        self.old_approx_kl = old_approx_kl
        self.approx_kl = approx_kl
        self.clip_fraction = clip_fraction
        self.explained_variance = explained_variance

class BasePpoAgent(nn.Module):
    def __init__(
        self, 
        base_network,
        actor_network,
        critic_network,
        observation_shape, 
        action_shape,
        learning_rate=2.5e-4,
        use_gpu=True
    ):
        """
        Initialize the PPO agent.

        Parameters:
            base_network (torch.nn.Module): The base network for feature extraction.
            actor_network (torch.nn.Module): The actor network.
            critic_network (torch.nn.Module): The critic network.
            observation_shape (tuple): The shape of the observation space.
            action_shape (tuple): The shape of the action space.
            learning_rate (float): The learning rate for the optimizer.
            use_gpu (bool): Whether to use GPU for training.
        """
        super(BasePpoAgent, self).__init__()

        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.learning_rate = learning_rate

        self.network = base_network
        self.actor = actor_network
        self.critic = critic_network

        # Creating optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

        # Choosing device
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
        elif torch.has_mps and use_gpu:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Moving torch module to chosen device
        self.to(self.device)

    def save_models(self, path: str):
        """
        Save the agent's models.

        Parameters:
            path (str): The path to the directory where models will be saved.
        """
        # Creating directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Saving network states
        torch.save(self.network.state_dict(), f"{path}/base.pth")
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_models(self, path: str):
        """
        Load the agent's models.

        Parameters:
            path (str): The path to the directory where models are saved.
        """
        # Checking if the models exist in the provided path
        assert os.path.exists(path), "Given path is invalid."
        assert os.path.isfile(f"{path}/base.pth"), "The given path does not contain the base model."
        assert os.path.isfile(f"{path}/actor.pth"), "The given path does not contain the actor model."
        assert os.path.isfile(f"{path}/critic.pth"), "The given path does not contain the critic model."
        
        # Loading models
        network_state_dict = torch.load(f"{path}/base.pth", map_location=self.device)
        actor_state_dict = torch.load(f"{path}/actor.pth", map_location=self.device)
        critic_state_dict = torch.load(f"{path}/critic.pth", map_location=self.device)

        # Updating networks with loaded models
        self.network.load_state_dict(network_state_dict)
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def forward(self, observations):
        """
        Compute the optimal action and corresponding value for a given observation.

        Parameters:
            observations (numpy.ndarray): The input observation(s) representing the state(s) of the environment.

        Returns:
            numpy.ndarray: The selected action(s) sampled from the action distribution.
            numpy.ndarray: The log probability of the selected action(s).
            numpy.ndarray: The entropy of the action distribution.
            numpy.ndarray: The estimated state value(s) for the input observation(s).
        """
        # Convert observations to a tensor on the appropriate device
        observations = torch.tensor(observations).to(self.device)

        # Calculate network outputs for policy (actor) and value function (critic)
        # without calculating gradients
        with torch.no_grad():
            hidden = self.network(observations / 255.0)
            logits = self.actor(hidden)
            value = self.critic(hidden)
        
        # Create a Categorical distribution over action logits
        probs = Categorical(logits=logits)

        # Sample actions, calculate log probabilities, and compute entropy
        actions = probs.sample()
        log_probs = probs.log_prob(actions)
        entropies = probs.entropy()

        # Convert tensors to NumPy arrays and return
        return actions.cpu().numpy(), log_probs.cpu().numpy(), entropies.cpu().numpy(), value.cpu().numpy()
    
    def _training_forward(self, observations: torch.Tensor, actions: torch.Tensor):
        """
        Compute the optimal action and corresponding value for a given observation.

        Parameters:
            observations (torch.Tensor): The input observation.
            actions (torch.Tensor): The action tensor (optional).

        Returns:
            torch.Tensor: The sampled action.
            torch.Tensor: The log probability of the action.
            torch.Tensor: The entropy of the action distribution.
            torch.Tensor: The value estimation.
        """
        hidden = self.network(observations / 255.0)
        logits = self.actor(hidden)
        value = self.critic(hidden)

        probs = Categorical(logits=logits)
        log_probs = probs.log_prob(actions)
        entropies = probs.entropy()

        return actions, log_probs, entropies, value
        
    def train(
            self,
            replay_buffer: ReplayBuffer, 
            gamma= 0.99, 
            enable_gae=True,
            gae_lambda=0.95, 
            clip_value_loss=True,
            clip_coef=0.1,
            max_norm_grad=0.5,
            value_coef=0.5, 
            entropy_coef=0.01, 
            lr_anneal_coef=None,
            target_kl=None,
            norm_adv=True,
            mini_batch_size=4,
            num_training_epochs=4):
        """
        Train the PPO agent.

        Parameters:
            replay_buffer (ReplayBuffer): The replay buffer containing the training data.
            gamma (float): The discount factor.
            enable_gae (bool): Whether to enable Generalized Advantage Estimation (GAE).
            gae_lambda (float): The lambda value for GAE.
            clip_value_loss (bool): Whether to clip the value loss.
            clip_coef (float): The clipping parameter.
            max_norm_grad (float): The maximum norm for gradient clipping.
            value_coef (float): The coefficient for the value loss.
            entropy_coef (float): The coefficient for the entropy loss.
            lr_anneal_coef (float): The coefficient for learning rate annealing.
            target_kl (float): The target KL divergence.
            norm_adv (bool): Whether to normalize advantages.
            mini_batch_size (int): The size of mini-batches.
            num_training_epochs (int): The number of training epochs.

        Returns:
            TrainingStats: The training statistics.
        """
        # Annealing the rate if instructed to do so.
        if lr_anneal_coef is not None:
            new_learning_rate = lr_anneal_coef * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = new_learning_rate

        # Convert replay buffer data into tensors
        observations = torch.Tensor(replay_buffer.processed_observations).to(self.device)
        actions = torch.Tensor(replay_buffer.actions).to(self.device)
        log_probs = torch.Tensor(replay_buffer.log_probs).to(self.device)
        rewards = torch.Tensor(replay_buffer.rewards).to(self.device)
        values = torch.Tensor(replay_buffer.values).to(self.device)
        terminations = torch.Tensor(replay_buffer.terminations).to(self.device)

        # Get the number of steps and the number of environments
        num_steps, num_envs = replay_buffer.num_steps_per_env, replay_buffer.num_envs

        if enable_gae:
            advantages = torch.zeros_like(rewards)
            last_gae_lambda = 0
            for t in reversed(range(num_steps - 1)):
                next_non_terminal = 1.0 - terminations[t + 1]
                next_values = values[t + 1]
                delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards)
            for t in reversed(range(num_steps - 1)):
                next_non_terminal = 1.0 - terminations[t + 1]
                next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * next_non_terminal * next_return
            advantages = returns - values

        # Flatten the batched transitions
        b_observations = observations.reshape((-1,) + self.observation_shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.action_shape)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        total_num_steps = num_steps * num_envs
        b_inds = np.arange(total_num_steps)
        clip_fractions = []
        for epoch in range(num_training_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, total_num_steps, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_value = self._training_forward(b_observations[mb_inds], b_actions.long()[mb_inds])
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fractions += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                if clip_value_loss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    value_loss = 0.5 * v_loss_max.mean()
                else:
                    value_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                # Backpropagation and optimization step
                entropy_loss = entropy.mean()
                loss = policy_loss - entropy_coef * entropy_loss + value_loss * value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm_grad)
                self.optimizer.step()

            # Early stopping based on KL divergence
            if target_kl is not None:
                if approx_kl > target_kl:
                    break
        
        # Compute training statistics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_variance = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        training_stats = TrainingStats(
            learning_rate=self.optimizer.param_groups[0]["lr"],
            value_loss=value_loss,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            old_approx_kl=old_approx_kl,
            approx_kl=approx_kl,
            clip_fraction=np.mean(clip_fractions),
            explained_variance=explained_variance
        )
        
        return training_stats
