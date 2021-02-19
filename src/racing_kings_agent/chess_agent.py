"""
Created on February 2021

@author: Andreas Spanopoulos

Self-playing Racing Kings Chess agent that interacts with the RacingKings chess environment.
"""


class RacingKingsChessAgent:
    """ Agent class """

    def __init__(self, env, nn):
        """ constructor """

        self.env = env
        self.nn = nn

    def play_episode(self):
        """ plays an episode of self-play """
        pass
