from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2
from vizdoom import DoomGame 

ACTION_SPACE_SIZE = 3
FRAMES_TO_SKIP = 4
# self.game.get_state().screen_buffer.shape

class DoomEnv(Env):
    def __init__(self, config_path, render=True):
        # Setting up game
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(render)
        self.game.init() # Start the game

        # Setting up observation and action space
        self.observation_space = Box(low=0, high=255, shape=(120, 160), dtype=np.uint8)
        self.action_space = Discrete(ACTION_SPACE_SIZE)

    def postprocess_frame(self, frame):
        # Modifying shape of frame to support opencv and converting frame grayscale 
        processed_frame = cv2.cvtColor(np.moveaxis(frame, 0, -1), cv2.COLOR_BGR2GRAY)
        # Resizing frame
        processed_frame = cv2.resize(processed_frame, (160, 120), interpolation=cv2.INTER_CUBIC)
        processed_frame = np.reshape(processed_frame, (120, 160)).astype(self.observation_space.dtype)

        return processed_frame
    
    def step(self, action):
        actions = np.identity(ACTION_SPACE_SIZE, dtype=np.uint8)
        reward = self.game.make_action(actions[action], FRAMES_TO_SKIP)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        if state: 
            frame = self.postprocess_frame(state.screen_buffer)
            info = { i: state.game_variables[i] for i in range(len(state.game_variables)) }
        else:
            frame = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            info = {  }

        return frame, reward, done, info
    
    def render(self):
        pass
    
    def reset(self):
        self.game.new_episode()
        return self.postprocess_frame(self.game.get_state().screen_buffer)
    
    def close(self):
        self.game.close()