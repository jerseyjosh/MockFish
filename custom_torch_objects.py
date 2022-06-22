import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.nn as nn
from config import *

class ChessDataset(Dataset):
    def __init__(self, root, path, transforms=Compose([ToTensor(), Normalize(0, 1)])):
        self.path = path
        self.transforms=transforms
        self.df = pd.read_pickle(DATA_DIR + DF_PATH) 

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
            # Rever toTensor HWC->CHW transformation as this is already in data
            board_state = self.transforms(board_state).permute((1, 2, 0)).contiguous()
        return board_state, from_square, to_square, piece_moved
    

class Mockfish(nn.Module):
    def __init__(self, numChannels, classes):
        super(Mockfish, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(in_features=24576, out_features=classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
