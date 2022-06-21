try:
    import cPickle as pickle
except:
    import pickle
import time
import gzip
import numpy as np
import pandas as pd
from config import *
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    
    def __init__(self, root, path, transforms=None):
        self.root = root
        self.path = path
        self.df = pd.read_pickle(DATA_DIR + DF_PATH)

    def __len__(self):
        return len(self.board_states)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        board_state = self.board_states[idx]
        leaving_square = self.leaving_squares[idx]
        return board_state, leaving_square


if __name__=="__main__":
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using MPS Acceleration: {torch.backends.mps.is_available()}")
    print(f"Using device: {DEVICE}")
    chessDataset = ChessDataset(DATA_DIR, f"training_data_{ELO_LOWER_LIMIT}.gzip")