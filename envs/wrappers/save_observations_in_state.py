import gymnasium as gym

class SaveObservationsInState(gym.Wrapper):
    def __init__(self, env: gym.Env, state: dict, state_key="observation"):
        super().__init__(env)
        self.env = env
        self.state = state
        self.state_key = state_key

    def reset(self, **kwargs):
        # Call the reset method of the wrapped environment
        observation, info = self.env.reset(**kwargs)

        # Save observation in state
        self.state[self.state_key] = observation

        return observation, info

    def step(self, action):
        # Perform the action and get the next state and other information from the environment
        observation, reward, done, truncated, info = self.env.step(action)

        # Save observation in state
        self.state[self.state_key] = observation

        # Return the next state and other information as usual
        return observation, reward, done, truncated, info