try:
    import cPickle as pickle
except:
    import pickle
import time
import gzip
import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
from custom_torch_objects import ChessDataset, Mockfish
from config import *



if __name__=="__main__":
    print("Loading dataset...")
    t0=time.time()
    chessData = ChessDataset(DATA_DIR, DF_PATH)
    t1=time.time()
    print(f"Took {t1-t0:.2f}s")

    mockfish = Mockfish(6, 64)
    optimizer = Adam(mockfish.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Generating train/test/validation split...")
    train_size = int(TRAIN_SIZE * len(chessData))
    valid_size = int(VALID_SIZE * len(chessData))
    test_size = len(chessData) - train_size - valid_size
    trainData, validData, testData = torch.utils.data.random_split(chessData, [train_size, valid_size, test_size])

    trainLoader = DataLoader(trainData, num_workers=0, batch_size=TRAIN_BATCH_SIZE)
    validLoader = DataLoader(validData, num_workers=0, batch_size=VALID_BATCH_SIZE)
    testLoader = DataLoader(testData, num_workers=0, batch_size=TEST_BATCH_SIZE)

    minvalid_loss = np.inf
    for x in range(EPOCHS):
        trainLoss = 0.0
        mockfish.train()     
        for data, label in trainLoader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            target = mockfish(data)
            loss = criterion(target,label)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()
        
        validLoss = 0.0
        mockfish.eval()    
        for data, label in validLoader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            target = mockfish(data)
            loss = criterion(target,label)
            validLoss = loss.item() * data.size(0)

        print(f'Epoch {x+1} \t\t Training data: {trainLoss / len(trainLoader)} \t\t Validation data: {validLoss / len(validLoader)}')

    