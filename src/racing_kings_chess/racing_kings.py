"""
Created on February 13 2021

In order for this script to work, you have to add the the python instruction 'global gameboard' to
line 56 of the chessboard.display.py file, so that it becomes:

def start(fen=''):
    global gameboard
    pygame.init()

You can also comment line 41:

def terminate():
    pygame.quit()
    # sys.exit()

if you want the terminate function to just close the display but not quit the execution.
"""

import torch
import chess.variant as variant
from chessboard import display
from time import sleep

from src.utils.error_utils import GameIsNotOverError
from src.racing_kings_chess.racing_kings_utils import *


class RacingKingsEnv:
    """ Wrapper class used to model the Racing Kings Chess environment """

    def __init__(self):
        self._board = variant.RacingKingsBoard()
        self.fen_white_pieces = ['N', 'B', 'R', 'Q', 'K']
        self.fen_black_pieces = ['n', 'b', 'r', 'q', 'k']
        self.repetitions_count = {piece_arrangement_from_fen(self._board.starting_fen): 1}

    @property
    def fen(self):
        """
        :return:  The FEN representation for the current state of the board.
        :rtype:   str
        """
        return self._board.fen()

    @property
    def legal_moves(self):
        """
        :return:  A list containing all the legal moves (as string) that can be played in the
                    current position (state).
        :rtype:   list(str)
        """
        return [self._board.uci(uci) for uci in list(self._board.legal_moves)]

    @property
    def is_finished(self):
        """
        :return:  True if the game has finished (loss/draw/win); Else False.
        :rtype:   Bool
        """
        return self._board.is_variant_end()

    @property
    def winner(self):
        """
        :return:  Returns 1 if white has won the game, -1 if black won or 0 if it was drawn.
        :rtype:   int

        :raises:
            GameIsNotOverError:  If the game has not ended, then this error is raised.
        """
        result = self._board.result()
        if result == '*':
            raise GameIsNotOverError
        return 1 if result == '1-0' else -1 if result == '0-1' else 0

    def reset(self):
        """
        :return:  None
        :rtype:   None

        Resets the board to the initial position.
        """
        self._board.reset()
        self.repetitions_count = {piece_arrangement_from_fen(self._board.starting_fen): 1}

    def set_fen(self, fen):
        """
        :param str fen:  The FEN string that describes a Racing Kings Chess position, which should
                            be set in the current environment.

        :return:  None
        :rtype:   None

        :raises:
            ValueError:  If the FEN provided is invalid for the Racing Kings Chess variant.

        Sets the current board position (state) to the one described by the input fen.
        """
        self._board.set_fen(fen)
        self.repetitions_count = {piece_arrangement_from_fen(fen): 1}

    def display_current_board(self, delay=3):
        """
        :param double delay:  Amount of seconds to display the board.

        :return:  None
        :rtype:   None

        Display the current position (state) of the board for a fixed amount of time
        (parameter: seconds).
        """
        display.start(self.fen)
        sleep(delay)
        display.terminate()

    def play_move(self, move):
        """
        :param str move:  The move to be played in the current position (state), e.g.: 'Nxc2'.

        :return:  None
        :rtype:   None

        Plays a specific move and updates the playing board accordingly.
        """
        self._board.push_san(move)

    @staticmethod
    def starting_fen():
        """
        :return:  The starting FEN position of a Racing Kings game.
        :rtype:   str
        """
        return variant.RacingKingsBoard().starting_fen

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

    def representation_of_starting_fen(self, t_history=8, n=8):
        """
        :param int t_history:  The length of the historic (previous) positions of each piece.
                                Used to make sure player does not fall in threefold repetition.
                                In the Alpha Zero paper, the default values is T = 8.
        :param int n:          Chess game dimension (should always be 8).

        # ToDo: explain the input, either here or in the README of the NeuralNetwork package.
        :return:  A list of 8 x 8 lists, used as input for the Neural Network.
        :rtype:   list[list[list]]

        Returns a list of 8 x 8 lists that will be used for input in the Neural Network.
        """

        # list to be returned, containing 99 8 x 8 board inputs
        cnn_input = []

        # white goes first
        players_pieces = [self.fen_white_pieces, self.fen_black_pieces]

        # for every piece of every player, compute its last 8 positions on the board
        for player_pieces in players_pieces:
            for piece in player_pieces:
                # if less than t_history moves have been played consider the previous values to be 0
                piece_history = [[[0 for _ in range(n)] for _ in range(n)]
                                 for _ in range(t_history - 1)]
                starting_config = fen_to_board_pieces(RacingKingsEnv.starting_fen(), piece, n=n)
                # in the starting fen of any variant, there are no previous positions so insert firs
                piece_history.insert(0, starting_config)

                cnn_input.extend(piece_history)

        # add the repetition planes for white: starting position has been reached once and others 0
        cnn_input.append(repetition_plane(1))
        cnn_input.extend([repetition_plane(0) for _ in range(t_history - 1)])
        # add the repetition planes for black: starting position has been reached once and others 0
        cnn_input.append(repetition_plane(1))
        cnn_input.extend([repetition_plane(0) for _ in range(t_history - 1)])

        # add the color board
        cnn_input.append(color_plane(1, n=n))

        # add the total moves plane: FEN positions start with this value set to 1
        cnn_input.append(total_moves_plane(1))

        # add the no progress count plane: FEN positions start with this value set to 0
        cnn_input.append(no_progress_count_plane(0))

        return cnn_input


env = RacingKingsEnv()
cnn_inp = torch.ByteTensor(env.representation_of_starting_fen())
torch.set_printoptions(threshold=10_000)
print(cnn_inp)
print(cnn_inp.shape)
