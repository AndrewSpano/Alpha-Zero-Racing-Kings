"""
Created on February 15 2021

@author: Andreas Spanopoulos

Utility function used for manipulation of chess boards and positions (states).
"""

import pprint


def piece_arrangement_from_fen(fen):
    """
    :param str fen:  The FEN string describing a position (state) in a Racing Kings Chess board.

    :return:  The first section of the FEN string describing the pieces arrangement on the board.
    :rtype:   str
    """
    return fen.split()[0]


def fen_to_board_pieces(fen, piece, n=8):
    """
    :param str fen:    The FEN string describing a position (state) in a Racing Kings Chess board.
    :param str piece:  Which piece type we want to find the positions.
    :param int n:      Chess game dimension (should always be 8).

    :return:  A list consisting of n sublists, each of length n, that depict and n x n grid, where
                each entry is 0, except for the specified pieces positions, that have value 1.
    :rtype:   list[list]
    """
    piece_placement, _, _, _, _, _ = fen.split()
    ranks = piece_placement.split('/')
    setup = [[0] * n for _ in range(n)]

    for rank_number, rank in enumerate(ranks):
        current_file = 0
        for info in rank:
            if info.isdigit():
                current_file += int(info)
            else:
                if info == piece:
                    setup[rank_number][current_file] = 1
                current_file += 1
    return setup


def repetition_plane(repetitions, n=8):
    """
    :param int repetitions:  Number of times a chess position (state) has been reached.
    :param int n:            Chess game dimension (should always be 8).

    :return:  An 8 x 8 list containing the same value for each entry, the repetitions number.
    :rtype:   list[list]

    This function computes the n x n repetitions plane.
    """
    return [[repetitions for _ in range(n)] for _ in range(n)]


def color_plane(color, n=8):
    """
    :param int color:  Integer value denoting the colour of a player (1 for white, 0 for black).
    :param int n:      Chess game dimension (should always be 8).

    :return:  An 8 x 8 list containing the same value for each entry, specified by the color.
    :rtype:   list[list]

    This function computes the n x n colour plane (1s for White, 0s for black).
    """
    return [[color for _ in range(n)] for _ in range(n)]


def total_moves_plane(moves, n=8):
    """
    :param int moves:  Integer value denoting the number of total moves played in the game.
    :param int n:      Chess game dimension (should always be 8).

    :return:  An 8 x 8 list containing the same value for each entry, the moves number.
    :rtype:   list[list]

    This function computes the n x n total moves plane.
    """
    return [[moves for _ in range(n)] for _ in range(n)]


def no_progress_count_plane(no_progress, n=8):
    """
    :param int no_progress:  Integer value denoting the number of total no-progress moves made.
    :param int n:            Chess game dimension (should always be 8).

    :return:  An 8 x 8 list containing the same value for each entry, the no_progress number.
    :rtype:   list[list]

    This function computes the n x n no progress count plane.
    """
    return [[no_progress for _ in range(n)] for _ in range(n)]


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)

