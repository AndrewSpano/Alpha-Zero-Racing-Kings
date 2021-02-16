"""
Created on February 15 2021

Utility function used for manipulation oh chess boards and positions.
"""

import chess.variant as variant
import torch
import pprint


def white_to_move(side_to_move, white_id='w'):
    """ wrapper that returns True if white is to move, else False """
    return side_to_move == white_id


def black_to_move(side_to_move, black_id='b'):
    """ wrapper that returns True if black is to move, else False """
    return side_to_move == black_id


def fen_to_board(fen, n=8):
    """ converts a FEN chess position to the appropriate input for the NN
        8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1 """
    # ignore castling and en passant as they can't occur in the Racing Kings variant
    piece_placement, side_to_move, _, _, halfmove_clock, fullmove_counter = fen.split()
    ranks = piece_placement.split('/')

    # account for player 1
    lst = [[0] * n for _ in range(n)]

    for rank_number, rank in enumerate(ranks):
        current_file = 0
        for info in rank:
            if info.isdigit():
                current_file += int(info)
            else:
                if info.isupper() and white_to_move(side_to_move):
                    lst[rank_number][current_file] = 1
                elif info.islower() and black_to_move(side_to_move):
                    lst[rank_number][current_file] = 1
                current_file += 1

    return lst


pp = pprint.PrettyPrinter(indent=4)

board = variant.RacingKingsBoard()
fen = board.fen()
lt = fen_to_board(fen)

# pp.pprint(lt)
