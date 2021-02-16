"""
Created on February 15 2021
"""

import unittest
import chess.variant as variant
from src.racing_kings_chess import racing_kings_utils


class TestRacingKingsUtils(unittest.TestCase):
    """ implements tests for the utility functions of the Racing Kings Chess variant """

    @classmethod
    def setUpClass(cls) -> None:
        # create here once the Racing Kings board
        cls.board = variant.RacingKingsBoard()
        cls.starting_fen = cls.board.starting_fen

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_fen_to_board(self):
        """ test the fen_to_board() function """
        from expected_values import fen_to_target_board
        for fen, target in fen_to_target_board.items():
            result = racing_kings_utils.fen_to_board(fen)
            self.assertEqual(result, target)
