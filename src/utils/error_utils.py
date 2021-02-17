"""
Created on February 17 2021

Contains custom Exceptions classes that can be used for debugging purposes.
"""


class GameIsNotOverError(Exception):
    """ Custom exception raised when the outcome of a game that has not yet finished is queried """

    def __init__(self, *args):
        """ constructor """
        self.fen = args[0] if args else None

    def __str__(self):
        """ print when raised outside try block """
        message = 'The game has not reached a terminal state.'
        if self.fen:
            message += f' The current FEN position is: {self.fen}'
        return message
