import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.nn as nn
import torch.nn.functional as F
from config import *
from sunfish.sunfish import N

class ChessDataset(Dataset):
    def __init__(self, root, path, target_piece, transforms=ToTensor()):
        self.path = path
        self.transforms=transforms
        self.df = pd.read_pickle(root + path).reset_index()
        if target_piece != 'selector':
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


'''
class Mockfish(nn.Module):
    def __init__(self):
        super(Mockfish, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        self.fc1 = nn.Linear(in_features=24576, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
'''

class Mockfish(nn.Module):
    def __init__(self, nhidden=1, hidden_size=256, dropout=False, dropout_rate=0.3):
        super(Mockfish, self).__init__()
        layers = [
            nn.Conv2d(in_channels=6, out_channels=96, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten()
        ]
        for _ in range(nhidden):
            if len(layers)==7:
                layers.append(nn.Linear(in_features=24576, out_features=hidden_size))
                if dropout:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
                if dropout:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=hidden_size, out_features=64))
        self.model = nn.Sequential(*layers)
        self.model.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)


    def forward(self, x):
        return self.model(x)


class MockfishBatchNorm(nn.Module):
    def __init__(self):
        super(MockfishBatchNorm, self).__init__()
        self.batchnorm1 = nn.BatchNorm2d(num_features=6)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(num_features=96)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.batchnorm3 = nn.BatchNorm2d(num_features=256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        self.batchnorm4 = nn.BatchNorm2d(num_features=384)
        self.fc1 = nn.Linear(in_features=24576, out_features=256)
        self.batchnorm5 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.batchnorm6 = nn.BatchNorm1d(num_features=64)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = F.relu(self.batchnorm2(self.conv1(x)))
        x = F.relu(self.batchnorm3(self.conv2(x)))
        x = F.relu(self.batchnorm4(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.batchnorm5(self.fc1(x)))
        x = self.batchnorm6(self.fc2(x))
        return x