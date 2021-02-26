"""
Created on February 26 2021

@author: Andreas Spanopoulos

Implements a class used as an interface between the actual moves of a Racing Kings chess game, and
how they are perceived as actions by a chess agent. It inherits the MoveTranslator() class.
"""


from src.environment.variants.actions.action_representations import MoveTranslator


class RacingKingsActions(MoveTranslator):
    """ interface between actual Racing Kings Chess moves and their corresponding IDs """

    def __init__(self):
        """ constructor """

        MoveTranslator.__init__(self,
                                files=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                                ranks=['1', '2', '3', '4', '5', '6', '7', '8'])

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
