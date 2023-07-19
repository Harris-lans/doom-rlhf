from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import cv2
from vizdoom import DoomGame

ACTION_SPACE_SIZE = 3  # Number of possible actions
FRAMES_TO_SKIP = 4  # Number of frames to skip per action

class DoomEnv(Env):
    """Custom environment for the Doom game."""
    
    def __init__(self, config_path, render=True):
        """
        Initialize the Doom game environment.

        Parameters:
            config_path (str): Path to the configuration file for the Doom game.
            render (bool): Whether to render the game window.
        """
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(render)
        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=self.game.get_state().screen_buffer.shape, dtype=np.uint8)
        self.action_space = Discrete(ACTION_SPACE_SIZE)

    def postprocess_frame(self, frame):
        """
        Preprocesses the frame by converting it to grayscale and resizing it.

        Parameters:
            frame (ndarray): The input game frame.

        Returns:
            ndarray: The preprocessed frame.
        """
        processed_frame = cv2.cvtColor(np.moveaxis(frame, 0, -1), cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.resize(processed_frame, (160, 120), interpolation=cv2.INTER_CUBIC)
        processed_frame = np.reshape(processed_frame, (120, 160)).astype(self.observation_space.dtype)

        return processed_frame
    
    def step(self, action):
        """
        Performs a game step by executing the chosen action.

        Parameters:
            action (int): The action to be taken.

        Returns:
            tuple: Tuple containing the preprocessed frame, reward, done flag, and additional information.
        """
        actions = np.identity(ACTION_SPACE_SIZE, dtype=np.uint8)
        reward = self.game.make_action(actions[action], FRAMES_TO_SKIP)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        if state:
            frame = self.postprocess_frame(state.screen_buffer)
            info = {i: state.game_variables[i] for i in range(len(state.game_variables))}
        else:
            frame = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            info = {}

        return frame, reward, done, info
    
    def render(self):
        """Dummy method for rendering the game. ViZDoom takes care of rendering"""
        pass
    
    def reset(self):
        """
        Resets the environment and starts a new episode.

        Returns:
            ndarray: The preprocessed initial frame.
        """
        self.game.new_episode()
        return self.postprocess_frame(self.game.get_state().screen_buffer)
    
    def close(self):
        """Closes the Doom game."""
        self.game.close()
