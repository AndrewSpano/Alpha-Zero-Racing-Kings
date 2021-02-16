"""
Created on February 13 2021

In order for this script to work, you have to add the the python instruction 'global gameboard' to
line 56 of the chessboard.display.py file, so that it becomes:

def start(fen=''):
    global gameboard
    pygame.init()

You can also comment line 41:

def terminate():
    pygame.quit()
    # sys.exit()

if you want the terminate function to just close the display but not quit the execution.
"""

import chess.variant as variant
from chessboard import display
from time import sleep

board = variant.RacingKingsBoard()
print(board.starting_fen)

moves = ['Kh3', 'Ka3',
         'Bd4', 'Ka4',
         'Kg4', 'Ka5',
         'Kh5', 'Ka6',
         'Rg7', 'Rb7',
         'Nxc2', 'Qxd4',
         'Rxb7', 'Rxb7',
         'Rg8', 'Ka7',
         'Ne2xd4', 'Rb8',
         'Kg6', 'Ba3',
         'Kg7', 'Rf8',
         'Rxf8', 'Bd6',
         'Kh8']

display.start(board.fen())
# while not display.checkForQuit():
while not board.is_variant_end():
    if moves:
        board.push_san(moves.pop(0))
        display.update(board.fen())
    sleep(0.5)

print(board.fen())
display.terminate()
