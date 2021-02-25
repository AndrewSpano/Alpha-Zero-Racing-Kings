"""
Created on February 13 2021

@author: Andreas Spanopoulos

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

import copy
import chess.variant as variant
from chessboard import display
from time import sleep

from src.utils.error_utils import GameIsNotOverError
from src.environment.racing_kings_utils import *


fen_white_pieces = ['N', 'B', 'R', 'Q', 'K']
fen_black_pieces = ['n', 'b', 'r', 'q', 'k']


class RacingKingsEnv:
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
        :param int n:                    Chess game dimension (should always be 8).

        Initialized the Racing Kings Chess environment. If an already initialized board and
        repetitions counts dictionary are passed, then it initializes it with these values.
        """
        self._board = _board or variant.RacingKingsBoard()
        self._repetitions_count = _repetitions_count or {self._board.board_fen(): 1}
        self._nn_input = _nn_input
        self.t_history = t_history
        self.n = n
        self._compute_current_state_representation()

    def copy(self):
        """
        :return:  A copy of the RacingKings Environment, with the same state as this one.
        :rtype:   RacingKingsEnv
        """
        return RacingKingsEnv(self._board.copy(), self._repetitions_count.copy(),
                              copy.deepcopy(self._nn_input), self.t_history, self.n)

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
        :return:  The starting FEN position of a Racing Kings game.
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

    def _compute_current_state_representation(self):
        """
        :return:  None
        :rtype:   None

        # ToDo: explain the input, either here or in the README of the NeuralNetwork package.
        Computes the list of 8 x 8 lists that will be used for input in the Neural Network.
        """

        # if this method is being called for the starting state
        if self.fen == self.starting_fen:

            # white goes first
            players_pieces = [fen_white_pieces, fen_black_pieces]

            self._nn_input = []

            # add the first input planes (note that self._nn_input is an empty list)
            self._nn_input.extend(starting_piece_setup(self.starting_fen, players_pieces,
                                                       self.t_history, self.n))

            # add the repetition planes for for past t_history positions reached twice
            self._nn_input.extend([repetition_plane(0, n=self.n) for _ in range(self.t_history)])
            # add the repetition planes for for past t_history positions reached once
            self._nn_input.extend([repetition_plane(1, n=self.n),
                                   *[repetition_plane(0, n=self.n)
                                     for _ in range(self.t_history - 1)]])

            # add the color plane: 1s for white, 0s for black
            self._nn_input.append(color_plane(1, n=self.n))

            # add the total moves plane: FEN positions start with this value set to 1
            self._nn_input.append(total_moves_plane(1, n=self.n))

            # add the no progress count plane: FEN positions start with this value set to 0
            self._nn_input.append(no_progress_count_plane(0, n=self.n))

        # if it is not being called for the starting state, use the previous state
        else:
            # determine which player is to move
            player_pieces = [fen_white_pieces, fen_black_pieces] if self._board.turn else \
                            [fen_black_pieces, fen_white_pieces]

            next_input = []
            # create the new setup from the previous position
            next_input.extend(current_piece_setup(self.fen, player_pieces, self._nn_input,
                                                  self.t_history, self.n))

            # add the repetitions planes
            repetitions_twice = 1 if self._repetitions_count[self._board.board_fen()] == 2 else 0
            repetitions_once = 1 if self._repetitions_count[self._board.board_fen()] == 1 else 0
            next_input.extend(update_repetitions_setup(repetitions_twice, repetitions_once,
                                                       player_pieces, self._nn_input,
                                                       self.t_history, self.n))

            # add the color plane
            next_input.append(color_plane(int(self._board.turn), n=self.n))

            # add the total moves plane and no progress count planes
            _, _, _, _, no_progress_count, total_moves = self.fen.split()
            next_input.append(total_moves_plane(int(total_moves), n=self.n))
            next_input.append(no_progress_count_plane(int(no_progress_count), n=self.n))

            self._nn_input = next_input

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
        'h6g7', 'a6b7',
        'g7f8', 'b7c8'
    ]

    while moves:
        # print(env._board.result())
        env.play_move(moves.pop(0))
        # print(env.is_finished)
        # pp.pprint(env.current_state_representation)
        # print('\n\n\n\n')

    # pp.pprint(env.legal_moves)
    #
    # from src.environment.action_representations import MoveTranslator
    #
    # mvt = MoveTranslator()
    #
    # s = mvt.get_move_ids_from_uci(env.legal_moves)
    #
    # for i in range(len(env.legal_moves)):
    #     print(f'{env.legal_moves[i]}: {s[i]}')

    # moves = [
    #     'h2h3', 'a2a3',
    #     'h3h2', 'a3a2',
    #     'h2h3', 'a2a3',
    #     'h3h2', 'a3a2'
    # ]
    #
    #
    # board = variant.RacingKingsBoard()
    #
    # while moves:
    #
    #     board.push_san(moves.pop(0))
    #
    #     print(f'Current repetitions == 1: {board.is_repetition(1)}')
    #     print(f'Current repetitions == 2: {board.is_repetition(2)}')
    #     print(f'Current repetitions == 3: {board.is_repetition(3)}')
    #     print(f'Regular fen: {board.fen()}')
    #     print(f'Board fen: {board.board_fen()}')
    #     print()
    #
    # b = board.copy()
    #
    # print(b.fen())
    # print(f'Current repetitions == 1: {b.is_repetition(1)}')
    # print(f'Current repetitions == 2: {b.is_repetition(2)}')
    # print(f'Current repetitions == 3: {b.is_repetition(3)}')
