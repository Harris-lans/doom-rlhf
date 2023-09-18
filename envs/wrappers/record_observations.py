import gym

class RecordObservations(gym.Wrapper):
    """
    A gym environment wrapper that records observations in the info dictionary.

    This wrapper stores the observations in the info dictionary under a specified key.

    Parameters:
        env (gym.Env): The wrapped gym environment.
        info_key (str): The key under which to store the observations in the info dictionary.
    """
    def __init__(self, env: gym.Env, info_key: str):
        super().__init__(env)
        self.env = env
        self.info_key = info_key

    def reset(self, **kwargs):
        """
        Reset the environment and record the initial observations in the info dictionary.

        Args:
            **kwargs: Additional keyword arguments passed to the reset method.

        Returns:
            tuple: A tuple containing the initial observations and info dictionary.
        """
        # Call the reset method of the wrapped environment
        observations, infos = self.env.reset(**kwargs)

        # Record the initial observations in the info dictionary
        infos[self.info_key] = observations

        return observations, infos

    def step(self, action):
        """
        Take a step in the environment, record observations in the info dictionary, and return the step result.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: A tuple containing the next observations, rewards, terminations, truncations, and the info dictionary.
        """
        # Perform the action and get the next state and other information from the environment
        observations, rewards, terminations, truncations, infos = self.env.step(action)

        # Record the observations in the info dictionary
        infos[self.info_key] = observations

        # Return the next state and other information as usual
        return observations, rewards, terminations, truncations, infos
