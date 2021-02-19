"""
Created on February 17 2021

@author: Andreas Spanopoulos

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


class InvalidConfigurationError(Exception):
    """ Custom Error raised when a configuration file is invalid """

    def __init__(self, **kwargs):
        """ constructor """
        super(InvalidConfigurationError, self).__init__()
        self.message = '' + kwargs.get('msg', '')

    def __str__(self):
        """ return string representation of the error """
        return self.message


class InvalidArchitectureError(Exception):
    """ Custom Error raised when the architecture of a Network is invalid """

    def __init__(self, **kwargs):
        """ constructor """
        super(InvalidArchitectureError, self).__init__()
        self.message = '' + kwargs.get('msg', '')

    def __str__(self):
        """ return string representation of the error """
        return self.message
