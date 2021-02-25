"""
Created on February 20 2021

@author: Andreas Spanopoulos

Module used to define the actions that can be taken in the Racing Kings environment, and provide
an interface between the actual actions and their corresponding IDs.
"""

import pprint


class MoveTranslator:
    """ interface between actual chess moves and their corresponding IDs """

    def __init__(self):
        """ constructor """

        self._files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self._ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
        self._size = (8, 8)
        lookups = self._build_square_coordinate_lookups()
        self._square_to_coordinate, self._coordinate_to_square = lookups
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

    def _build_id_and_move_lookups(self):
        """
        :return:  Dictionaries that map moves to their respective IDs and vice-versa.
        :rtype:   tuple(dict, dict)

        The dictionaries look like this:  move_to_id['a1a2'] = 0,
                                          coordinate_to_square[1] = 'a1a3'
        """
        move_id = 0
        move_to_id = {}
        id_to_move = {}

        for coordinate, square in self._coordinate_to_square.items():

            squares_reachable_by_queen_move = self._squares_queens_move_away(coordinate)
            squares_reachable_by_knight_move = self._squares_knights_move_away(coordinate)
            reachable_squares = squares_reachable_by_queen_move + squares_reachable_by_knight_move

            for reachable_square in reachable_squares:

                move = square + reachable_square
                move_to_id[move] = move_id
                id_to_move[move_id] = move

                move_id += 1

        return move_to_id, id_to_move

    @property
    def num_actions(self):
        """
        :return:  The number of different actions that exist.
        :rtype:   int
        """
        return len(self._move_to_id)

    def get_move_id(self, move):
        """
        :param str move:  A Racing Kings Chess move.

        :return:  The ID of a Racing Kings Chess move, as compute by the method
                    _build_id_and_move_lookups()
        :rtype:   int
        """
        return self._move_to_id[move]

    def get_move(self, _id):
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


if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    m = MoveTranslator()

    # pp.pprint(m.move_to_id)
    # print(len(m.id_to_move))

    # print(m.get_move(69))
