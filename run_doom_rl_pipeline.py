import time
import argparse
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import gymnasium as gym
from utils.os import clear_console
from utils.env import make_vizdoom_env
from utils.replay_buffer import ReplayBuffer
from agents.doom_ppo_agent import DoomPpoAgent
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

def parse_args():
    # Environment and training specific arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-cfg", type=str, default=None,
        help="The path to the ViZDoom config file")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
        help="The learning rate of the optimizer")
    parser.add_argument("--total-timesteps", type=int, default=300000,
        help="Total timesteps to train for")
    parser.add_argument("--render-env", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="If toggled, one environment will be rendered")
    parser.add_argument("--model-save-threshold", type=int, default=7000,
        help="Number of steps before model saving is enabled")
    parser.add_argument("--enable-gpu", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="If toggled, gpu will be used for training and using agent")
    parser.add_argument("--track-stats", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="If toggled, the script will record training statistics using ")

    # Agent specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="The number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="The number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--enable-gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="The discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="The lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="The number of mini-batches")
    parser.add_argument("--training-epochs", type=int, default=10,
        help="The number of training epochs when training the agent")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="The surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
        help="Coefficient of the entropy")
    parser.add_argument("--value-coef", type=float, default=0.5,
        help="Coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="The maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="The target KL divergence threshold")
    
    args = parser.parse_args()
   
    return args

if __name__ == "__main__":
    # Parsing command line args
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        # Storing start time for saving models and logs with timestamp
        start_datetime = datetime.now()
        start_datetime_timestamp_str = start_datetime.strftime('%Y_%m_%d_%H_%M_%S')

        # Setting up agent training config
        global_step = 0
        start_time = start_datetime.timestamp()
        batch_size = int(args.num_envs * args.num_steps)
        minibatch_size = batch_size // args.num_minibatches
        num_updates = args.total_timesteps // batch_size

        # Initializing environments
        envs = gym.vector.SyncVectorEnv([ make_vizdoom_env(args.env_cfg, render_mode="human" if i == 0 and args.render_env else None) for i in range(args.num_envs)])

        # Setting up agent
        agent = DoomPpoAgent(envs.single_observation_space, 
                            envs.single_action_space,
                            learning_rate=args.learning_rate,
                            use_gpu=args.enable_gpu)

        # Creating replay buffer for storing transitions
        replay_buffer = ReplayBuffer(args.num_steps, 
                                    args.num_envs, 
                                    envs.envs[0].raw_observation_space, 
                                    envs.single_observation_space, 
                                    envs.single_action_space)

        if args.track_stats:
            tensorboard_logs_directory = f"./stats/doom_basic_level/rl_training_{start_datetime_timestamp_str}"

            # Setting up debugging for Tensorboard
            tensorboard_writer = SummaryWriter(tensorboard_logs_directory)
            logger.info(f"Saving tensorboard data to {tensorboard_logs_directory}")

        # Preparing environments and tracking variables for training 
        observations, infos = envs.reset()
        terminations = [ 0 for _ in range(args.num_envs) ]
        best_average_return = float('-inf')
        returns = []
        clear_console()

        for update in tqdm(range(1, num_updates + 1), desc ="Training Doom PPO Agent", colour="#4287f5"):
            if args.anneal_lr:
                # Calculating learning rate annealing coefficient
                learning_rate_anneal_coef = 1.0 - (update - 1.0) / num_updates
            else:
                learning_rate_anneal_coef = None

            for step in tqdm(range(0, args.num_steps), desc ="Exploring the environment", colour="#42f551", leave=False):
                global_step += args.num_envs

                # Getting next action and it's value
                actions, log_probs, probs, values = agent.forward(observations)
                values = values.flatten()

                observations_, rewards, terminations_, truncations, infos = envs.step(actions)

                # Saving transitions in replay buffer
                replay_buffer[step] = (
                    np.stack(infos["raw_observations"]),
                    observations,
                    actions,
                    log_probs,
                    rewards,
                    values,
                    terminations
                )

                # Saving new observation and termination status for next step
                observations = observations_
                terminations =  terminations_
                
                if 'final_info' in infos:
                    for env_info in infos['final_info']:
                        if env_info is not None and "episode" in env_info.keys():
                            logger.info(f"global_step={global_step}, episodic_return={env_info['episode']['r']}")

                            # Recording returns
                            returns.append(env_info['episode']['r'])

                            # Writing environment stats to TensorBoard
                            if args.track_stats:
                                tensorboard_writer.add_scalar("ppo_agent/charts/episodic_return", env_info["episode"]["r"], global_step)
                                tensorboard_writer.add_scalar("ppo_agent/charts/episodic_length", env_info["episode"]["l"], global_step)

                            break

            # Calculating current mean episodic return
            current_mean_episodic_return = np.mean(returns)
            logger.info(f"Current Mean Episodic Return = {current_mean_episodic_return}")

            # Checking if the current mean is higher than previous highest mean 
            # and if the number of steps taken exceeds the model save threshold, and then saving the model
            if current_mean_episodic_return > best_average_return and global_step > args.model_save_threshold:
                # Saving the model
                model_save_path = f"./models/rl_pipeline/training_run_{start_datetime_timestamp_str}/doom_ppo_agent/checkpoint_step_{global_step}"
                logger.info(f"Saving models to `{model_save_path}`...")
                agent.save_models(model_save_path)
                logger.info(f"Successfully saved models to `{model_save_path}`!")

                # Saving new best average return and clearing returns arrays
                best_average_return = current_mean_episodic_return
                returns.clear()
            
            # Training the agent
            training_stats = agent.train(
                replay_buffer=replay_buffer,
                gamma=args.gamma,
                enable_gae=args.enable_gae,
                gae_lambda=args.gae_lambda,
                clip_vloss=args.clip_vloss,
                clip_coef=args.clip_coef,
                max_grad_norm=args.max_grad_norm,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                learning_rate_anneal_coef=learning_rate_anneal_coef,
                target_kl=args.target_kl,
                normalize_advantages=args.norm_adv,
                mini_batch_size=minibatch_size,
                num_training_epochs=args.training_epochs
            )

            # Writing training stats to TensorBoard
            if args.track_stats:
                tensorboard_writer.add_scalar("ppo_agent/charts/learning_rate", training_stats.learning_rate, global_step)
                tensorboard_writer.add_scalar("ppo_agent/losses/value_loss", training_stats.value_loss, global_step)
                tensorboard_writer.add_scalar("ppo_agent/losses/policy_loss", training_stats.policy_loss, global_step)
                tensorboard_writer.add_scalar("ppo_agent/losses/entropy_loss", training_stats.entropy_loss, global_step)
                tensorboard_writer.add_scalar("ppo_agent/charts/old_approx_kl", training_stats.old_approx_kl, global_step)
                tensorboard_writer.add_scalar("ppo_agent/charts/approx_kl", training_stats.approx_kl, global_step)
                tensorboard_writer.add_scalar("ppo_agent/charts/clip_fraction", training_stats.clip_fraction, global_step)
                tensorboard_writer.add_scalar("ppo_agent/charts/explained_variance", training_stats.explained_variance, global_step)
                tensorboard_writer.add_scalar("ppo_agent/charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Closing environments
        envs.close()

        if args.track_stats:
            # Closing tensorboard writer
            tensorboard_writer.close()