"""
Created on February 27 2021

@author: Andreas Spanopoulos

Implements utility functions for the Dataset classes.
"""

import os
import time
import random
import logging
import chess.pgn

from collections import deque
from src.environment.variants.base_chess_env import ChessEnv
from src.environment.actions.action_representations import MoveTranslator
from src.utils.main_utils import pad_to_maxlen


def parse_pgn_game(game, env, mvt, min_white_elo, min_black_elo, is_worse_game=False):
    """
    :param chess.pgn.Game game:  The game to be parsed.
    :param ChessEnv env:         The Chess environment that will be used to parse these games.
    :param MoveTranslator mvt:   The Move Translator that will be used to parse these games.
    :param int min_white_elo:    The minimum Elo rating that a player with the white pieces can have
                                    in order to be considered an expert, and therefore take into
                                    account his moves.
    :param int min_black_elo:    The minimum Elo rating that a player with the black pieces can have
                                    in order to be considered an expert, and therefore take into
                                    account his moves.
    :param int is_worse_game:    Whether this game is not an expert game, but we want to sample all
                                    the moves anyway in order to show actual checkmates (higher
                                    ranks usually resign before checkmates).

    :return:  A list containing tuples of the form:
               (legal_actions_padded, state_representation, game_outcome (z), observed_actions (pi))
    :rtype:   tuple[list[int], list[list[list[int]]], int, int]
    """

    # list containing the data observations that will be gathered from this game
    game_data = []

    # get info about the game
    result = game.headers['Result']
    white_elo = int(game.headers['WhiteElo'])
    black_elo = int(game.headers['BlackElo'])

    # reset the environment
    env.reset()

    # play out the game in order to store the states and legal actions
    for idx, uci_move_object in enumerate(game.mainline_moves()):

        # get the move as a string
        move = str(uci_move_object)

        # if white is to play and has enough elo, or black is to play and has enough elo
        if (idx % 2 == 0 and white_elo > min_white_elo) or \
           (idx % 2 == 1 and black_elo > min_black_elo) or is_worse_game:

            # legal moves and state representation
            legal_actions = mvt.get_move_ids_from_uci(env.legal_moves)
            pad_to_maxlen(legal_actions, maxlen=mvt.legal_moves_upper_bound)
            st = env.current_state_representation

            # result was drawn
            if result == '1/2-1/2':
                z = 0
            # white is playing and white won, or black is playing and black won
            elif (idx % 2 == 0 and result == '1-0') or (idx % 2 == 1 and result == '0-1'):
                z = 1
            # white is playing and white lost, or black is playing and black lost
            else:
                z = -1

            # get the "optimal action" which the player played, basically cloning his behaviour
            pi = mvt.id_from_move(move)

            # add the current instance of data to the observed data
            game_data.append((legal_actions, st, z, pi))

        # whatever happens (state is added to the deque or not), play the move
        env.play_move(move)

    return game_data


def parse_data(root_directory, env, mvt, deque_maxlen, min_white_elo=2000, min_black_elo=2000,
               worse_games=15):
    """
    :param str root_directory:  The directory containing (only) the database games in .pgn format.
    :param ChessEnv env:        The Chess environment that will be used to parse these games.
    :param int deque_maxlen:    The maximum length that the deque (replay buffer) can have.
    :param MoveTranslator mvt:  The Move Translator that will be used to parse these games.
    :param int min_white_elo:   The minimum Elo rating that a player with the white pieces can have
                                    in order to be considered an expert, and therefore take into
                                    account his moves.
    :param int min_black_elo:   The minimum Elo rating that a player with the black pieces can have
                                    in order to be considered an expert, and therefore take into
                                    account his moves.
    :param int worse_games:     Number of games to sample per file, that are not made by experts.

    :return:  A deque containing the information parsed from the games in the .pgn files inside
                the root directory.
    :rtype:   deque

    Note that to download all the games you can use the bash script "download_racing_kings_data.sh".
    """

    # data deque to be returned
    data = deque([], maxlen=deque_maxlen)

    # for each data file in the root directory
    for filename in os.listdir(root_directory):

        # construct the full path, read the file and start scanning the games
        full_path = os.path.join(root_directory, filename)
        pgn = open(full_path)
        game = chess.pgn.read_game(pgn)

        # log information
        start_time = time.time()
        logging.info(f'Starting to parse data file {full_path}.')

        # how many games "bad" games have been considered so far for this file
        worse_games_considered = 0

        # while there are still games to be read from the file
        while game is not None:

            # get the elo ratings of the players
            white_elo = int(game.headers['WhiteElo'])
            black_elo = int(game.headers['BlackElo'])

            # If any of the players is an "expert"
            if white_elo >= min_white_elo or black_elo >= min_black_elo:
                # parse data from the game
                data += parse_pgn_game(game, env, mvt, min_white_elo, min_black_elo, False)
            # else if the game is not made by experts, but we want to parse it anyway
            elif worse_games_considered < worse_games and random.uniform(0, 1) < 0.5:
                data += parse_pgn_game(game, env, mvt, min_white_elo, min_black_elo, True)
                worse_games_considered += 1

            # read the next game
            game = chess.pgn.read_game(pgn)

        logging.info(f'After parsing file {full_path}, the number of training examples are: '
                     f'{len(data)}. The time it took to parse the data file is: '
                     f'{time.time() - start_time:.2f}s\n')

    logging.info('All files have been parsed.\n\n')
    return data
