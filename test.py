from utils.env import make_doom_env
from torch.utils.tensorboard import SummaryWriter
from agents.ppo_agent import PpoAgent
from utils.memory import Memory
from utils.time import current_timestamp_ms
import torch
import time
import gym
import numpy as np

num_envs = 1

# Initializing environment
envs = gym.vector.SyncVectorEnv(
    [make_doom_env(level_config_path='vizdoom/scenarios/basic.cfg') for i in range(num_envs)]
)

# Setting up agent
agent = PpoAgent(envs.single_observation_space, envs.single_action_space)

# Setting up agent training config
global_step = 0
start_time = time.time()
num_steps = 128
num_mini_batches = 4
batch_size = int(num_envs * num_steps)
mini_batch_size = int(batch_size / num_mini_batches)
total_timesteps = 10000000
num_updates = int(total_timesteps / batch_size)
memory = Memory(agent.device, num_steps, num_envs, envs.single_observation_space.shape, envs.single_action_space.shape)

# Setting up debugging for Tensorboard
tensorboard_writer = SummaryWriter(f"runs/run_{current_timestamp_ms()}")

observation = torch.Tensor(envs.reset()).to(agent.device)
done = torch.zeros(num_envs).to(agent.device)

for update in range(1, num_updates + 1):
    # Calculating learning rate annealing coefficient
    learning_rate_anneal_coef = 1.0 - (update - 1.0) / num_updates

    for step in range(0, num_steps):
        global_step += num_envs

        # Getting next action and it's value
        with torch.no_grad():
            action, log_prob, _, value = agent.get_optimal_action_and_value(observation)
            value = value.flatten()

        observation_, reward, done_, info = envs.step(action.cpu().numpy())

        # Saving experience in memory
        memory.remember(
            step=step, 
            observation= observation,
            action=action,
            value=value,
            log_prob=log_prob,
            reward=torch.tensor(np.array(reward, dtype=np.float32)).to(agent.device).view(-1),
            done=done
        )

        # Saving new observation and done status for next step
        observation = torch.Tensor(observation_).to(agent.device) 
        done =  torch.Tensor(done_).to(agent.device)

        # Writing step debug info to TensorBoard
        for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    tensorboard_writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    tensorboard_writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

    training_stats = agent.train(
        memory.observations,
        memory.actions,
        memory.log_probs,
        memory.rewards,
        memory.values,
        memory.dones,
        batch_size
    )

    print("SPS:", int(global_step / (time.time() - start_time)))

    tensorboard_writer.add_scalar("charts/learning_rate", training_stats.learning_rate, global_step)
    tensorboard_writer.add_scalar("losses/value_loss", training_stats.value_loss, global_step)
    tensorboard_writer.add_scalar("losses/policy_loss", training_stats.policy_loss, global_step)
    tensorboard_writer.add_scalar("losses/entropy_loss", training_stats.entropy_loss, global_step)
    tensorboard_writer.add_scalar("charts/old_approx_kl", training_stats.old_approx_kl, global_step)
    tensorboard_writer.add_scalar("charts/approx_kl", training_stats.approx_kl, global_step)
    tensorboard_writer.add_scalar("charts/clip_fraction", training_stats.clip_fraction, global_step)
    tensorboard_writer.add_scalar("charts/explained_variance", training_stats.explained_variance, global_step)
    tensorboard_writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
