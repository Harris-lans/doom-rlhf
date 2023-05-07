import unittest

from vizdoom import DoomGame 

class VizDoomInitializationTest(unittest.TestCase):

    def test(self):
        success = True

        try:
            game = DoomGame()
            game.load_config('vizdoom/scenarios/basic.cfg')
            game.init()

            game.close()
        except:
            success = False

        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()