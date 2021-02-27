"""
Created on February 20 2021

@author: Andreas Spanopoulos

Module used to define the actions that can be taken in a Chess environment, and provide an interface
between the actual actions and their corresponding IDs.
"""

from abc import abstractmethod


class MoveTranslator:
    """ interface between actual chess moves and their corresponding IDs """

    def __init__(self, files, ranks):
        """
        :param list[str] files:       The files of the chess board.
        :param list[str] ranks:       The ranks of the chess board.
        """

        self._files = files
        self._ranks = ranks
        self._size = (len(ranks), len(files))
        self._square_to_coordinate, self._coordinate_to_square = \
            self._build_square_coordinate_lookups()
        self._move_to_id, self._id_to_move = self._build_id_and_move_lookups()

    def _build_square_coordinate_lookups(self):
        """
        :return:  Dictionaries that map squares to their respective coordinates on the board
                    and vice-versa.
        :rtype:   tuple(dict, dict)

        The dictionaries look like this:  square_to_coordinate['a1'] = (0, 0),
                                          coordinate_to_square[(0, 1)] = 'a2'
        """
        square_to_coordinate = {}
        coordinate_to_square = {}
        for file_index, file in enumerate(self._files):
            for rank_index, rank in enumerate(self._ranks):
                square_id, square = (file_index, rank_index), file + rank
                square_to_coordinate[square] = square_id
                coordinate_to_square[square_id] = square

        return square_to_coordinate, coordinate_to_square

    def _is_within_board(self, coordinate):
        """
        :param tuple coordinate:  The coordinates of a possible square in the chess board.

        :return:  True if the square if legal (within boundaries); Else False
        :rtype:   bool
        """
        return 0 <= coordinate[0] < self._size[0] and 0 <= coordinate[1] < self._size[1]

    def _squares_queens_move_away(self, coordinate):
        """
        :param tuple coordinate:  The coordinates of a square in the chess board.

        :return:  A list of squares that are accessible with a single Queen move.
        :rtype:   list[str]
        """
        queen_moves = []
        offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        for offset in offsets:
            target = (coordinate[0] + offset[0], coordinate[1] + offset[1])
            while self._is_within_board(target):
                queen_moves.append(self._coordinate_to_square[target])
                target = (target[0] + offset[0], target[1] + offset[1])

        return queen_moves

    def _squares_knights_move_away(self, coordinate):
        """
        :param tuple coordinate:  The coordinates of a square in the chess board.

        :return:  A list of squares that are accessible with a single Knight move.
        :rtype:   list[str]
        """
        offsets = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
        knight_jumps = map(lambda off: (coordinate[0] + off[0], coordinate[1] + off[1]), offsets)
        return [self._coordinate_to_square[knight_jump] for knight_jump in knight_jumps
                if self._is_within_board(knight_jump)]

    @abstractmethod
    def _build_id_and_move_lookups(self):
        """
        :return:  Dictionaries that map moves to their respective IDs and vice-versa.
        :rtype:   tuple(dict, dict)

        The dictionaries look like this:  move_to_id['a1a2'] = 0,
                                          coordinate_to_square[1] = 'a1a3'
        """
        pass

    @property
    def num_actions(self):
        """
        :return:  The number of different actions that exist.
        :rtype:   int
        """
        return len(self._move_to_id)

    @property
    @abstractmethod
    def legal_moves_upper_bound(self):
        """
        :return:  The max number of legal moves that a player may have at any state, for the
                    corresponding chess variant.
        :rtype:   int
        """
        pass

    def id_from_move(self, move):
        """
        :param str move:  A Racing Kings Chess move.

        :return:  The ID of a Racing Kings Chess move, as compute by the method
                    _build_id_and_move_lookups()
        :rtype:   int
        """
        return self._move_to_id[move]

    def move_from_id(self, _id):
        """
        :param int _id:  An ID representing a Racing Kings Chess move.

        :return:  The corresponding move as computed by the method _build_id_and_move_lookups()
        :rtype:   str
        """
        return self._id_to_move[_id]

    def get_move_ids_from_uci(self, uci_moves):
        """
        :param list[str] uci_moves:  A list containing Chess moves in UCI notation.

        :return:  A list containing the IDs of each move respectively.
        :rtype:   list[int]
        """
        return [self._move_to_id[move] for move in uci_moves]

    def get_ucis_from_move_ids(self, ids):
        """
        :param list[int] ids:  A list containing chess move IDs.

        :return:  A list with the respective UCI moves.
        :rtype:   list[str]
        """
        return [self._id_to_move[_id] for _id in ids]


if __name__ == "__main__":
    pass
