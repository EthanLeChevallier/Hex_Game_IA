import unittest

from board_hex import BoardHex
from game_state_hex import GameStateHex
from player_hex import PlayerHex
from seahorse.game.light_action import LightAction


class GameStateHexTests(unittest.TestCase):
    def setUp(self):
        self.red = PlayerHex("R", name="Red")
        self.blue = PlayerHex("B", name="Blue")
        players = [self.red, self.blue]
        scores = {player.get_id(): 0 for player in players}
        self.state = GameStateHex(
            scores=scores,
            next_player=self.red,
            players=players,
            rep=BoardHex(env={}, dim=[2, 2]),
            step=0,
        )

    def test_initial_state_has_one_action_per_cell(self):
        self.assertEqual(len(list(self.state.generate_possible_light_actions())), 4)

    def test_red_wins_with_a_vertical_connection(self):
        self.state = self.state.apply_action(
            LightAction({"piece": "R", "position": (0, 0)})
        )
        self.state = self.state.apply_action(
            LightAction({"piece": "B", "position": (0, 1)})
        )
        self.state = self.state.apply_action(
            LightAction({"piece": "R", "position": (1, 0)})
        )

        self.assertTrue(self.state.is_done())
        self.assertEqual(self.state.scores[self.red.get_id()], 1.0)
        self.assertEqual(self.state.scores[self.blue.get_id()], 0.0)


if __name__ == "__main__":
    unittest.main()