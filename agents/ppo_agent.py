import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
from utils.time import *

def layer_init(layer, std=np.sqrt(2), bias_constant=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_constant)
    return layer

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
        self.learning_rate = learning_rate 
        self.value_loss = value_loss
        self.policy_loss = policy_loss
        self.entropy_loss = entropy_loss
        self.old_approx_kl = old_approx_kl
        self.approx_kl = approx_kl
        self.clip_fraction = clip_fraction
        self.explained_variance = explained_variance

class PpoAgent(nn.Module):
    def __init__(
        self, 
        observation_space, 
        action_space, 
        actor_model_path=None, 
        critic_model_path=None,
        learning_rate=2.5e-4,
        use_gpu=True
    ):
        super(PpoAgent, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate

        # Creating model
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self.observation_space.shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 11 * 16, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, self.action_space.n if isinstance(self.action_space, Discrete) else self.action_space), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

         # Choosing the device to run agent on
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
        elif torch.has_mps and use_gpu:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Running the agent on the device
        self.to(self.device)

    def get_optimal_action_and_value(self, observation, action=None):
        hidden = self.network(observation / 255.0)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        
        probs = Categorical(logits=logits)
        action = probs.sample() if action is None else action
        log_probs = probs.log_prob(action)
        probs = probs.entropy()

        return action, log_probs, probs, value

    def train(
            self,
            next_observation,
            next_done,
            observations, 
            actions, 
            log_probs, 
            rewards, 
            values, 
            dones,
            num_steps,
            batch_size, 
            gamma= 0.99, 
            enable_gae=True,
            gae_lambda=0.95, 
            clip_vloss=True,
            epsilon=0.1,
            max_grad_norm=0.5,
            value_coef=0.5, 
            entropy_coef=0.01, 
            learning_rate_anneal_coef=None,
            target_kl=None,
            normalize_advantages=True,
            mini_batch_size=4,
            num_training_epochs=4):
        
        # Annealing the rate if instructed to do so.
        if learning_rate_anneal_coef is not None:
            lrnow = learning_rate_anneal_coef * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            next_value = self.critic(self.network(next_observation / 255.0)).reshape(1, -1)

            if enable_gae:
                advantages = torch.zeros_like(rewards).to(self.device)
                last_gae_lambda = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                    advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(self.device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * next_non_terminal * next_return
                advantages = returns - values
        
        # Flatten the batch
        b_observations = observations.reshape((-1,) + self.observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.action_space.shape)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clip_fractions = []
        for epoch in range(num_training_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_value = self.get_optimal_action_and_value(b_observations[mb_inds], b_actions.long()[mb_inds])
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fractions += [((ratio - 1.0).abs() > epsilon).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds],
                        -epsilon,
                        epsilon,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    value_loss = 0.5 * v_loss_max.mean()
                else:
                    value_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - entropy_coef * entropy_loss + value_loss * value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

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
