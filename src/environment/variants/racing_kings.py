"""
Created on February 13 2021

@author: Andreas Spanopoulos

Implements the Racing Kings Environment, which inherits from the Base Chess environment.
"""

import copy
import chess.variant as variant

from chessboard import display
from time import sleep

from src.environment.variants.chess_utils import *
from src.environment.variants.base_chess_env import ChessEnv

fen_white_pieces = ['N', 'B', 'R', 'Q', 'K']
fen_black_pieces = ['n', 'b', 'r', 'q', 'k']


class RacingKingsEnv(ChessEnv):
    """ Wrapper class used to model the Racing Kings Chess environment """

    def __init__(self, _board=None, _repetitions_count=None, _nn_input=None, t_history=8, n=8):
        """
        :param BoardT _board:            A Chess board object containing an already initialized
                                            game.
        :param dict _repetitions_count:  The dictionary containing the number of repetitions
                                            per position.
        :param list _nn_input:           List containing the most current state representation of
                                            the board.
        :param int t_history:            The length of the historic (previous) positions of the
                                            board to store. Used to take threefold repetition into
                                            account. In the paper, the default values is T = 8.
        :param int n:                    Chess game dimension (usually 8).

        Initializes the Racing Kings Chess environment. If an already initialized board and
        repetitions counts dictionary are passed, then it initializes it with these values.
        """
        ChessEnv.__init__(self,
                          _board=_board or variant.RacingKingsBoard(),
                          _repetitions_count=_repetitions_count,
                          _nn_input=_nn_input,
                          t_history=t_history,
                          n=n)

    def copy(self):
        """
        :return:  A copy of the RacingKings Environment, with the same state as this one.
        :rtype:   RacingKingsEnv
        """
        return RacingKingsEnv(self._board.copy(), self._repetitions_count.copy(),
                              copy.deepcopy(self._nn_input), self._t_history, self._n)

    def _compute_current_state_representation(self):
        """
        :return:  None
        :rtype:   None

        # ToDo: explain the input, either here or in the README of the NeuralNetwork package.
        Computes the list of n x n lists that will be used for input in the Neural Network.
        """

        # if this method is being called for the starting state
        if self.fen == self.starting_fen:

            # white goes first
            players_pieces = [fen_white_pieces, fen_black_pieces]

            self._nn_input = []

            # add the first input planes (note that self._nn_input is an empty list)
            self._nn_input.extend(starting_piece_setup(self.starting_fen, players_pieces,
                                                       self._t_history, self._n))

            # add the repetition planes for for past t_history positions reached twice
            self._nn_input.extend([repetition_plane(0, n=self._n) for _ in range(self._t_history)])
            # add the repetition planes for for past t_history positions reached once
            self._nn_input.extend([repetition_plane(1, n=self._n),
                                   *[repetition_plane(0, n=self._n)
                                     for _ in range(self._t_history - 1)]])

            # add the color plane: 1s for white, 0s for black
            self._nn_input.append(color_plane(1, n=self._n))

            # add the total moves plane: FEN positions start with this value set to 1
            self._nn_input.append(total_moves_plane(1, n=self._n))

            # add the no progress count plane: FEN positions start with this value set to 0
            self._nn_input.append(no_progress_count_plane(0, n=self._n))

        # if it is not being called for the starting state, use the previous state
        else:
            # determine which player is to move
            player_pieces = [fen_white_pieces, fen_black_pieces] if self._board.turn else \
                [fen_black_pieces, fen_white_pieces]

            next_input = []
            # create the new setup from the previous position
            next_input.extend(current_piece_setup(self.fen, player_pieces, self._nn_input,
                                                  self._t_history, self._n))

            # add the repetitions planes
            repetitions_twice = 1 if self._repetitions_count[self._board.board_fen()] == 2 else 0
            repetitions_once = 1 if self._repetitions_count[self._board.board_fen()] == 1 else 0
            next_input.extend(update_repetitions_setup(repetitions_twice, repetitions_once,
                                                       player_pieces, self._nn_input,
                                                       self._t_history, self._n))

            # add the color plane
            next_input.append(color_plane(int(self._board.turn), n=self._n))

            # add the total moves plane and no progress count planes
            _, _, _, _, no_progress_count, total_moves = self.fen.split()
            next_input.append(total_moves_plane(int(total_moves), n=self._n))
            next_input.append(no_progress_count_plane(int(no_progress_count), n=self._n))

            self._nn_input = next_input

    @staticmethod
    def simulate_game(ply_list, ply_delay=0.5):
        """
        :param list ply_list:     List containing the plies (individual moves) of each player,
                                    sequentially. E.g.: ['Kh3', 'Ka3', 'Bd4', 'Ka4', ...]
        :param double ply_delay:  Delay time between plies shown in the display.

        :return:  The FEN configuration of the last position.
        :rtype:   str

        Simulates a game from a given ply list, displaying the moves along the way. Then, the last
        FEN position is returned.
        """
        _board = variant.RacingKingsBoard()
        plies = ply_list.copy()

        display.start(_board.fen())
        while not _board.is_variant_end() and not display.checkForQuit():
            if plies:
                _board.push_san(plies.pop(0))
                display.update(_board.fen())
            sleep(ply_delay)
        display.terminate()

        return _board.fen()


if __name__ == "__main__":

    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    env = RacingKingsEnv()
    # pp.pprint(env.current_state_representation)

    moves = [
        'h2h3', 'a2a3',
        'h3h4', 'a3a4',
        'h4h5', 'a4a5',
        'h5h6', 'a5a6',
        'h6g7', 'a6a5',
        'g7f8', 'a5a6'
    ]

    while moves and not env.is_finished:
        # print(env._board.result())
        env.play_move(moves.pop(0))
        # if not env.is_finished:
        #     print(f'move {moves[0]} is terminal: {env.is_terminal_move(moves[0])}')
        # print(env.is_finished)
        # print(env.is_finished)
        # pp.pprint(env.current_state_representation)
        # print('\n\n\n\n')
