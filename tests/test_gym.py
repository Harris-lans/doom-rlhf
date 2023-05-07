import unittest

import gym

class GymDefaultEnvironmentTest(unittest.TestCase):

    def test(self):
        success = True

        try:
          env = gym.make("LunarLander-v2", render_mode="human")
          env.action_space.seed(42)

          observation, info = env.reset(seed=42)

          for _ in range(1000):
              observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

              if terminated or truncated:
                  observation, info = env.reset()

          env.close()
        except:
          success = False

        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()