import gymnasium as gym

class RecordObservations(gym.Wrapper):
    def __init__(self, env: gym.Env, info_key: str):
        super().__init__(env)
        self.env = env
        self.info_key = info_key

    def reset(self, **kwargs):
        # Call the reset method of the wrapped environment
        observations, infos = self.env.reset(**kwargs)

        infos[self.info_key] = observations

        return observations, infos

    def step(self, action):
        # Perform the action and get the next state and other information from the environment
        observations, rewards, terminations, truncations, infos = self.env.step(action)

        infos[self.info_key] = observations

        # Return the next state and other information as usual
        return observations, rewards, terminations, truncations, infos