"""
Created on February 15 2021

@author: Andreas Spanopoulos
"""

import unittest
from src.utils.error_utils import GameIsNotOverError
from src.environment.variants.racing_kings import RacingKingsEnv


class TestRacingKingsEnv(unittest.TestCase):
    """ implements tests for the utility functions of the Racing Kings Chess variant """

    @classmethod
    def setUpClass(cls) -> None:
        # create here once the Racing Kings board
        cls.env = RacingKingsEnv()

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        self.env.reset()

    def tearDown(self) -> None:
        pass

    def test_starting_fen(self):
        """ test the starting_fen() method of the Racing Kings Environment class """
        from expected_values import starting_fen
        self.assertEqual(self.env.starting_fen, starting_fen)

    def test_legal_moves(self):
        """ test the legal_moves property of the Racing Kings Environment class """
        from expected_values import moves_to_legal_moves
        for moves, target_legal_moves in moves_to_legal_moves.items():
            self.env.reset()
            for move in moves:
                self.env.play_move(move)
            self.assertEqual(self.env.legal_moves, target_legal_moves)

    def test_fen(self):
        """ test the fen method of the Racing Kings Environment class """
        from expected_values import moves_to_fen
        for moves, target_fen in moves_to_fen.items():
            self.env.reset()
            for move in moves:
                self.env.play_move(move)
            self.assertEqual(self.env.fen, target_fen)

    def test_is_finished(self):
        """ test the is_finished property of the Racing Kings Environment class """
        from expected_values import moves_to_is_finished
        for moves, target_is_finished in moves_to_is_finished.items():
            self.env.reset()
            for move in moves:
                self.env.play_move(move)
            self.assertEqual(self.env.is_finished, target_is_finished)

    def test_winner(self):
        """ test the winner property of the Racing Kings Environment class """
        from expected_values import moves_to_winner
        for moves, target_winner in moves_to_winner.items():
            self.env.reset()
            for move in moves:
                self.env.play_move(move)

            try:
                winner = self.env.winner
                self.assertEqual(winner, target_winner)
            except GameIsNotOverError:
                if target_winner != '*':
                    self.assertTrue(False)

    def test_representation_of_starting_fen(self):
        """ test the representation_of_starting_fen() method of the Racing Kings Environment
            class """
        from expected_values import starting_fen_input
        self.assertEqual(self.env.current_state_representation, starting_fen_input)
