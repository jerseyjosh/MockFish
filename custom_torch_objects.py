import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
from config import *

class ChessDataset(Dataset):
    def __init__(self, root, path, transforms=[ToTensor]):
        self.root = root
        self.path = path
        self.df = pd.read_pickle(DATA_DIR + DF_PATH)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        board_state = self.df.board_states[idx]
        from_square = self.df.from_squares[idx]
        to_square = self.df.to_squares[idx]
        return board_state, from_square, to_square
    


class Mockfish(nn.Module):
    def __init__(self, numChannels, classes):
        super(Mockfish, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = nn.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output
    
