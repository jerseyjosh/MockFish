import pickle
import time
import gzip
import numpy as np
from config import *
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    
    def __init__(self, root, board_states, from_squares, to_squares, transforms=None):
        with gzip.open(root + board_states, "rb") as f:
            self.board_states = pickle.load(f)
        with open(root + from_squares, "rb") as f:
            self.from_squares = pickle.load(f)
        return
    
    def __len__(self):
        return len(self.board_states)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        board_state = self.board_states[idx]
        leaving_square = self.leaving_squares[idx]
        return board_state, leaving_square
    




if __name__=="__main__":
    t0=time.time()
    with gzip.open(DATA_DIR + f"training_data_{ELO_LOWER_LIMIT}.gzip", "rb") as f:
        training_data = pickle.load(f)
    print(training_data.head())
    t1 = time.time()
    print(f"took {t1-t0}s")
    #DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    #print(f"Using MPS Acceleration: {torch.backends.mps.is_available()}")
    #chessDataset = ChessDataset(DATA_DIR, "board_states.gzip", "leaving_squares.gzip")