"""
Created on February 26 2021

@author: Andreas Spanopoulos

Defines base class for Chess environments and their basic operations.


Note for chessboard display:

In order for the display to work, you have to add the the python instruction 'global gameboard' to
line 56 of the chessboard.display.py file, so that it becomes:

def start(fen=''):
    global gameboard <----
    pygame.init()

You can also comment line 41:

def terminate():
    pygame.quit()
    # sys.exit()

if you want the terminate function to just close the display but not quit the execution.
"""

import copy

from abc import abstractmethod
from chessboard import display
from time import sleep

from src.utils.error_utils import GameIsNotOverError


class ChessEnv:
    """ Base class used to define a chess environment """

    def __init__(self, _board, _repetitions_count, _nn_input, t_history, n):
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
        :param int n:                    Chess game dimension (should always be 8).
        """
        self._board = _board
        self._repetitions_count = _repetitions_count or {self._board.board_fen(): 1}
        self._nn_input = _nn_input
        self._t_history = t_history
        self._n = n
        self._compute_current_state_representation()

    @abstractmethod
    def copy(self):
        """
        :return:  A copy of the current environment.
        :rtype:   ChessEnv
        """
        pass

    @property
    def current_state_representation(self):
        """
        :return:  A list of 8 x 8 lists, used as input for the Neural Network.
        :rtype:   list[list[list[int]]
        """
        return copy.deepcopy(self._nn_input)

    @property
    def starting_fen(self):
        """
        :return:  The starting FEN position of the chess variant that inherits from this base class.
        :rtype:   str
        """
        return self._board.starting_fen

    @property
    def fen(self):
        """
        :return:  The FEN representation for the current state of the board.
        :rtype:   str
        """
        return self._board.fen()

    @property
    def side_to_move(self):
        """
        :return:  'w' or 'b', depending on whether white or black is to play the next move.
        :rtype:   str
        """
        _, side, _, _, _, _ = self.fen.split()
        return side

    @property
    def moves(self):
        """
        :return:  The number of moves played in the game thus far.
        :rtype:   int
        """
        _, _, _, _, _, total_moves = self.fen.split()
        return int(total_moves)

    @property
    def legal_moves(self):
        """
        :return:  A list containing all the legal moves (as strings) that can be played in the
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
        return self._repetitions_count[self._board.board_fen()] == 3 or self._board.is_variant_end()

    @property
    def winner(self):
        """
        :return:  Returns 1 if white has won the game, -1 if black won or 0 if it was drawn.
        :rtype:   int

        :raises:
            GameIsNotOverError:  If the game has not ended, then this error is raised.
        """
        if self._repetitions_count[self._board.board_fen()] == 3:
            return 0
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
        self._repetitions_count = {self._board.board_fen(): 1}
        self._compute_current_state_representation()

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
        self.reset()
        self._board.set_fen(fen)
        self._repetitions_count = {self._board.board_fen(): 1}

    def play_move(self, move):
        """
        :param str move:  The move to be played in the current position (state), e.g.: 'Nxc2'.

        :return:  None
        :rtype:   None

        Plays a specific move and updates the playing board accordingly.
        """
        self._board.push_san(move)
        board_fen = self._board.board_fen()
        self._repetitions_count[board_fen] = self._repetitions_count.get(board_fen, 0) + 1
        self._compute_current_state_representation()

    @abstractmethod
    def _compute_current_state_representation(self):
        """
        :return:  None
        :rtype:   None

        Computes the list of 8 x 8 lists that will be used for input in the Neural Network.
        """
        pass

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

    @staticmethod
    @abstractmethod
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
        pass
