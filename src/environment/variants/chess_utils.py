"""
Created on February 15 2021

@author: Andreas Spanopoulos

Utility function used for manipulation of chess boards and positions (states).
"""


def fen_to_board_pieces(fen, piece, n=8):
    """
    :param str fen:    The FEN string describing a position (state) in a Racing Kings Chess board.
    :param str piece:  Which piece type we want to find the positions. E.g. 'K', 'k', 'B', 'b'
    :param int n:      Chess game dimension (should always be 8).

    :return:  A list consisting of n sublists, each of length n, that depict and n x n grid, where
                each entry is 0, except for the specified pieces positions, that have value 1.
    :rtype:   list[list[int]]
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


def starting_piece_setup(starting_fen, players_pieces, t_history=8, n=8):
    """
    :param str starting_fen:     The starting fen of the Racing Kings Chess variant.
    :param list players_pieces:  List containing a list with the pieces of each player respectively.
    :param int t_history:        The length of the historic (previous) positions of each piece.
    :param int n:                Chess game dimension (should always be 8).

    :return:  A list of lists of lists containing the representation of the starting board state.
    :rtype:   list[list[list[int]]]
    """

    setup = []
    # for every piece of every player, compute its last T positions on the board
    for player_pieces in players_pieces:
        for piece in player_pieces:
            # if less than t_history moves have been played consider the previous values to be 0
            piece_history = [[[0 for _ in range(n)] for _ in range(n)]
                             for _ in range(t_history - 1)]
            starting_config = fen_to_board_pieces(starting_fen, piece, n=n)
            # in the starting fen of any variant, there are no previous positions so insert it first
            piece_history.insert(0, starting_config)

            setup.extend(piece_history)

    return setup


def current_piece_setup(fen, players_pieces, previous_setup, t_history=8, n=8):
    """
    :param str fen:              The current fen of the Racing Kings Chess variant.
    :param list players_pieces:  List containing a list with the pieces of each player respectively.
    :param list previous_setup:  The list containing the full previous setup of the state.
    :param int t_history:        The length of the historic (previous) positions of each piece.
    :param int n:                Chess game dimension (usually 8).

    :return:  A list of lists of lists containing the representation of the current board state.
    :rtype:   list[list[list[int]]]
    """

    opponent_num_pieces = len(players_pieces[1])
    setup = []
    # for every piece of the current player, take its current position and (t_history - 1) previous
    for idx, piece in enumerate(players_pieces[0]):
        # compute current position of the piece, as it may have been removed due to the last move
        piece_config = fen_to_board_pieces(fen, piece, n)
        # the T previous opponent setup planes become the planes of the current player
        piece_history = [piece_config,
                         *previous_setup[(opponent_num_pieces + idx) * t_history:
                                         (opponent_num_pieces + 1 + idx) * t_history - 1]]
        setup.extend(piece_history)

    # for every piece of the opponent player, take its current position and (t_history - 1) previous
    for idx, piece in enumerate(players_pieces[1]):
        # compute current position of the piece, as it may have changed due to the last move
        piece_config = fen_to_board_pieces(fen, piece, n)
        # add the past (t_history - 1) historic positions for that piece
        piece_history = [piece_config, *previous_setup[idx * t_history: (idx + 1) * t_history - 1]]
        setup.extend(piece_history)

    return setup


def repetition_plane(repetitions, n=8):
    """
    :param int repetitions:  Number of times a chess position (state) has been reached.
    :param int n:            Chess game dimension (usually 8).

    :return:  An n x n list containing the same value for each entry, the repetitions number.
    :rtype:   list[list[int]]

    This function computes the n x n repetitions plane.
    """
    return [[repetitions for _ in range(n)] for _ in range(n)]


def color_plane(color, n=8):
    """
    :param int color:  Integer value denoting the colour of a player (1 for white, 0 for black).
    :param int n:      Chess game dimension (usually 8).

    :return:  An n x n list containing the same value for each entry, specified by the color.
    :rtype:   list[list[int]]

    This function computes the n x n colour plane (1s for White, 0s for black).
    """
    return [[color for _ in range(n)] for _ in range(n)]


def total_moves_plane(moves, n=8):
    """
    :param int moves:  Integer value denoting the number of total moves played in the game.
    :param int n:      Chess game dimension (should always be 8).

    :return:  An 8 x 8 list containing the same value for each entry, the moves number.
    :rtype:   list[list[int]]

    This function computes the n x n total moves plane.
    """
    return [[moves for _ in range(n)] for _ in range(n)]


def no_progress_count_plane(no_progress, n=8):
    """
    :param int no_progress:  Integer value denoting the number of total no-progress moves made.
    :param int n:            Chess game dimension (usually 8).

    :return:  An n x n list containing the same value for each entry, the no_progress number.
    :rtype:   list[list[int]]

    This function computes the n x n no progress count plane.
    """
    return [[no_progress for _ in range(n)] for _ in range(n)]


def update_repetitions_setup(repetitions_twice, repetitions_once, players_pieces, previous_setup,
                             t_history=8, n=8):
    """
    :param int repetitions_twice:  1 if current position has been repeated exactly twice. Else 0.
    :param int repetitions_once:   1 if current position has been repeated exactly once; Else 0.
    :param list players_pieces:    List containing lists with the pieces of each player.
    :param list previous_setup:    The list containing the full previous setup of the state.
    :param int t_history:          The length of the historic (previous) positions of each piece.
    :param int n:                  Chess game dimension (usually 8).

    :return:  A list of lists of lists containing the repetition planes of the current board state.
    :rtype:   list[list[list[int]]]
    """

    num_pieces = len(players_pieces[0]) + len(players_pieces[1])
    setup = []

    # get the history for whether the previous positions where repeated exactly 2 times
    repetitions_history = previous_setup[num_pieces * t_history: (num_pieces + 1) * t_history - 1]
    # add the current plane for whether this position has been reached exactly 2 times
    setup.extend([repetition_plane(repetitions_twice, n=n), *repetitions_history])

    # get the history for whether the previous positions where repeated once
    repetitions_history = previous_setup[(num_pieces + 1) * t_history:
                                         (num_pieces + 2) * t_history - 1]
    # add the current plane for whether this position has been reached exactly one time
    setup.extend([repetition_plane(repetitions_once, n=n), *repetitions_history])

    return setup


if __name__ == "__main__":
    pass
