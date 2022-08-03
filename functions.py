import numpy as np
import pandas as pd
import chess
import re
import os
import time
from config import *
from custom_torch_objects import ChessDataset
from torch.utils.data import DataLoader

# removes extraneous symbols
def parse_san(moves):
    parsed_moves = re.sub(r'(\d+\.\s)|(x)|(#)|(\+)', lambda x: '', moves)
    return parsed_moves

# converts to long algebraic notation: e4 -> e2e4 etc.
def san_to_lan(moves, fen=None):
    if fen is None:
        board = chess.Board()
        fen = board.fen()
    board = chess.Board(fen=fen)
    lan = ""
    for move in moves.split():
        # exception to catch illegal moves
        try:
            lan += str(board.push_san(move)) + " "
        except ValueError:
            return None
    return lan[:-1]


# converts fen representation to board object
def fen_to_board(fen, piece_values=PIECE_VALUES, white_turn=True):
    board = np.zeros((6, 8, 8), dtype="float32")
    new_fen = ''
    for char in fen.split(' ')[0]: # edited to accomodate full FEN
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


def get_model_path(dir, piece):
    pattern = f"_{piece}_"
    for f in os.listdir(dir):
        if re.search(pattern, f):
            return dir + f

# create dataloaders for specific pieces
def create_dataloaders(target_piece, dir, path):
    print(f"Loading {target_piece} data...")
    print(f"loading dataloader: {path}")
    data = ChessDataset(dir, path, target_piece=target_piece)
    dataLoader = DataLoader(data, num_workers=NUM_WORKERS, batch_size=TRAIN_BATCH_SIZE)
    return dataLoader


# adapted from sklearn implementation
def confusion_matrix(y_true, y_pred, N=64):
    #N = max(max(y_true), max(y_pred)) + 1
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    return torch.sparse.LongTensor(
        torch.stack([y_true, y_pred]), 
        torch.ones_like(y_true, dtype=torch.long),
        torch.Size([N, N])).to_dense()