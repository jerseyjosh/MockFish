import numpy as np
import pandas as pd
import chess
import re
from config import *

# removes extraneous symbols
def parse_san(moves):
    parsed_moves = re.sub(r'(\d+\.\s)|(x)|(#)|(\+)', lambda x: '', moves)
    return parsed_moves

# converts to long algebraic notation: e4 -> e2e4 etc.
def san_to_lan(moves):
  board = chess.Board()
  lan = ""
  for move in moves.split():
    lan += str(board.push_san(move)) + " "
  return lan[:-1]

# Generate 6x8x8 board array
def init_board(piece_values=PIECE_VALUES): 
    INIT_PAWNS = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [-piece_values[0] for i in range(8)],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [piece_values[0] for i in range(8)],
                        [0, 0, 0, 0, 0, 0, 0, 0]])

    INIT_KNIGHTS = np.array([[0, -piece_values[1], 0, 0, 0, 0, -piece_values[1], 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, piece_values[1], 0, 0, 0, 0, piece_values[1], 0]])

    INIT_BISHOPS = np.array([[0, 0, -piece_values[2], 0, 0, -piece_values[2], 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, piece_values[2], 0, 0, piece_values[2], 0, 0]])

    INIT_ROOKS = np.array([[-piece_values[3], 0, 0, 0, 0, 0, 0, -piece_values[3]],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [piece_values[3], 0, 0, 0, 0, 0, 0, piece_values[3]]])

    INIT_QUEENS = np.array([[0, 0, 0, -piece_values[4], 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, piece_values[4], 0, 0, 0, 0]])

    INIT_KINGS = np.array([[0, 0, 0, 0, -piece_values[5], 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, piece_values[5], 0, 0, 0]])

    INIT_BOARD = np.stack((INIT_PAWNS, INIT_KNIGHTS, INIT_BISHOPS, INIT_ROOKS, INIT_QUEENS, INIT_KINGS))
    return INIT_BOARD

def fen_to_board(fen, piece_values=PIECE_VALUES, white_turn=True):
    board = np.zeros((6, 8, 8), dtype="float32")
    new_fen = ''
    for char in fen:
        if char.isnumeric():
            new_fen += "0" * int(char)
        else:
            new_fen += char
    for yidx, row in enumerate(new_fen.split('/')):
        for xidx, char in enumerate(row):
            if char=='p':
                board[0, yidx, xidx] = -piece_values[0]
            elif char=="P":
                board[0, yidx, xidx] = piece_values[0]
            elif char=="n":
                board[1, yidx, xidx] = -piece_values[1]
            elif char=="N":
                board[1, yidx, xidx] = piece_values[1]
            elif char=="b":
                board[2, yidx, xidx] = -piece_values[2]
            elif char=="B":
                board[2, yidx, xidx] = piece_values[2]
            elif char=="r":
                board[3, yidx, xidx] = -piece_values[3]
            elif char=="R":
                board[3, yidx, xidx] = piece_values[3]
            elif char=="q":
                board[4, yidx, xidx] = -piece_values[4]
            elif char=="Q":
                board[4, yidx, xidx] = piece_values[4]
            elif char=="k":
                board[5, yidx, xidx] = -piece_values[5]
            elif char=="K":
                board[5, yidx, xidx] = piece_values[5]
            else:
                pass
    if not white_turn:
        return -np.flip(board, axis=(1, 2))
    else:
        return board

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)