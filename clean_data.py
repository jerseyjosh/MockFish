import pandas as pd
import numpy as np
import os
import chess
import re
from config import *

def load_data(root, path, elo_limit):
    chess_data = pd.read_csv(DATA_DIR + "2016_CvC.csv")
    chess_data = chess_data[((chess_data["White Elo"] > ELO_LOWER_LIMIT)&(chess_data["Black Elo"]> ELO_LOWER_LIMIT))]
    return chess_data

def clean_data(data):
    moves = chess_data.Moves
    parsed_sans = moves.apply(lambda x: parse_san(x))
    parsed_lans = parsed_sans.apply(lambda x: san_to_lan(x))
    return parsed_sans, parsed_lans

def parse_san(moves):
    parsed_moves = re.sub(r'(\d+\.\s)|(x)|(#)|(\+)', lambda x: '', moves)
    return parsed_moves

def san_to_lan(moves):
  board = chess.Board()
  lan = ""
  for move in moves.split():
    lan += str(board.push_san(move)) + " "
  return lan[:-1]

if __name__=="__main__":
    chess_data = load_data(DATA_DIR, "2016_CvC.csv", ELO_LOWER_LIMIT)
    parsed_sans, parsed_lans = clean_data(chess_data)
    if not os.path.isfile(DATA_DIR + f"parsed_lans_{ELO_LOWER_LIMIT}elo.zip"):
        parsed_lans.to_csv(DATA_DIR + f"parsed_lans_{ELO_LOWER_LIMIT}elo.zip")
    if not os.path.isfile(DATA_DIR + f"parsed_sans_{ELO_LOWER_LIMIT}elo.zip"):
        parsed_sans.to_csv(DATA_DIR + f"parsed_sans_{ELO_LOWER_LIMIT}elo.zip")