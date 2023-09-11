import time
import argparse
from distutils.util import strtobool

import numpy as np
import gymnasium as gym
from utils.env import make_vizdoom_env
from agents.doom_ppo_agent import DoomPpoAgent

import logging
from tqdm import tqdm
from utils.os import clear_console
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Environment arguments
    parser.add_argument("--env-cfg", type=str, default=None,
        help="The path to the ViZDoom scenario config file for the environment")
    # Agent arguments
    parser.add_argument("--agent-model", type=str, default=None,
        help="The path to the trained model for the agent")
    parser.add_argument("--enable-gpu", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="If toggled, gpu will be used by agent")
    # Play arguments
    parser.add_argument("--episodes", type=int, default=50,
        help="Number of episodes to play")
    
    args = parser.parse_args()
   
    return args

if __name__ == "__main__":
    # Parsing command line args
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        # Creating Environment
        num_envs = 1
        envs = gym.vector.SyncVectorEnv([ make_vizdoom_env(level=args.env_cfg, render_mode="human") for i in range(num_envs)])

        # Setting up agent
        agent = DoomPpoAgent(envs.single_observation_space, 
                             envs.single_action_space,
                             use_gpu=args.enable_gpu)
        agent.load_models(path=args.agent_model)

        number_of_episodes = args.episodes
        total_rewards = 0

        clear_console()

        for episode in tqdm(range(number_of_episodes), desc ="Playing DOOM!", colour="#4287f5", leave=False):
            observations, infos = envs.reset()
            terminations = np.zeros(num_envs, dtype=np.int32)
            total_episode_rewards = 0

            while not terminations.all():
                # Getting next action and it's value
                action, log_prob, _, value = agent.forward(observations)
                observations, rewards, terminations, truncations, infos = envs.step(action)
                total_episode_rewards += rewards

                time.sleep(1/30)

            logger.info(f"Total reward for episode {episode} is {total_episode_rewards}")
            total_rewards += total_episode_rewards

            time.sleep(0.25)

        logger.info(f"Average reward per episode is {total_rewards / number_of_episodes}")

        envs.close()