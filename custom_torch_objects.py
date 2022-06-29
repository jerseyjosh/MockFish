import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.nn as nn
import torch.nn.functional as F
from config import *

class ChessDataset(Dataset):
    def __init__(self, root, path, transforms=ToTensor(), target_piece=None):
        self.path = path
        self.transforms=transforms
        self.df = pd.read_pickle(DATA_DIR + DF_PATH) 
        if target_piece is not None:
            self.df = self.df[self.df.pieces_moved.str.lower()==target_piece].reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        board_state = self.df.board_states[idx]
        from_square = self.df.from_squares[idx]
        to_square = self.df.to_squares[idx]
        piece_moved = self.df.pieces_moved[idx]
        if self.transforms:
            # Reverse toTensor HWC->CHW transformation as this is already in data
            board_state = self.transforms(board_state).permute((1, 2, 0)).contiguous()
        return board_state, from_square, to_square, piece_moved


# no weight initialisation -> 34% piece selector accuracy
class MockfishBaseline(nn.Module):
    def __init__(self, numChannels, classes, init_weights=None):
        super(MockfishBaseline, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(in_features=24576, out_features=256)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(in_features=256, out_features=classes)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

