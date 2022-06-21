import pandas as pd
import numpy as np
import os
import sys
import pickle
import gzip
import chess
from tqdm import tqdm
import re
import time
from functions import *
from config import *

def load_data(root, path, elo_limit):
    chess_data = pd.read_csv(DATA_DIR + "2016_CvC.csv")
    chess_data = chess_data[((chess_data["White Elo"] > ELO_LOWER_LIMIT)&(chess_data["Black Elo"]> ELO_LOWER_LIMIT))]
    return chess_data

def clean_data(data, save_path):
    moves = data.Moves
    parsed_sans = moves.apply(lambda x: parse_san(x))
    if not os.path.isfile(save_path + f"parsed_lans_{ELO_LOWER_LIMIT}elo.zip"):
        parsed_lans = parsed_sans.apply(lambda x: san_to_lan(x))
        print("Saving parsed_lans.csv...")
        parsed_lans.to_csv(save_path + f"parsed_lans_{ELO_LOWER_LIMIT}elo.zip", header=False, index=False)
    else:
        parsed_lans = pd.read_csv(save_path + f"parsed_lans_{ELO_LOWER_LIMIT}elo.zip")
    return parsed_sans, parsed_lans


# SAN/LAN input to [board_states], [next_moves]
def generate_training_data(games, save_path):
    board_states = []
    from_squares = []
    to_squares = []
    for game in tqdm(games):
        board = chess.Board()
        white_turn=True
        for move in game.split():
            from_square = chess.parse_square(move[:2])
            to_square = chess.parse_square(move[2:4])
            piece_moved = board.piece_at(from_square)
            board_array = fen_to_board(board.board_fen(), piece_values=PIECE_VALUES, white_turn=white_turn)
            board_states.append(board_array)
            from_squares.append(from_square)
            to_squares.append(str(piece_moved) + str(to_square))
            board.push_san(move)
            white_turn = not white_turn

    df = pd.DataFrame({"board_states": board_states, "from_squares": from_squares, "to_squares": to_squares})

    if not os.path.isfile(DATA_DIR + f"training_data_{ELO_LOWER_LIMIT}.h5"):
        t0 = time.time()
        print("Saving HDF...")
        df.to_hdf(DATA_DIR + f"training_data_{ELO_LOWER_LIMIT}.csv")
        t1 = time.time()
        print(f"Took {t1-t0}s")

    return df


if __name__=="__main__":
    print("Loading data...")
    chess_data = load_data(DATA_DIR, "2016_CvC.csv", ELO_LOWER_LIMIT)
    print("Parsing games...")
    parsed_sans, parsed_lans = clean_data(chess_data, DATA_DIR)
    print(len(parsed_lans[0]))
    #print("Generating training data...")
    #training_data = generate_training_data(parsed_lans, DATA_DIR)
    #print("Done.")
    