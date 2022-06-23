import pickle
from venv import create
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
from custom_torch_objects import ChessDataset, Mockfish
from config import *

def create_dataloaders():
    print("Loading dataset...")
    t0=time.time()
    chessData = ChessDataset(DATA_DIR, DF_PATH)
    t1=time.time()
    print(f"Took {t1-t0:.2f}s")

    print("Generating train/test/validation split...")
    train_size = int(TRAIN_SIZE * len(chessData))
    valid_size = int(VALID_SIZE * len(chessData))
    test_size = len(chessData) - train_size - valid_size
    trainData, validData, testData = torch.utils.data.random_split(chessData, [train_size, valid_size, test_size])

    trainLoader = DataLoader(trainData, num_workers=NUM_WORKERS, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validLoader = DataLoader(validData, num_workers=NUM_WORKERS, batch_size=VALID_BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(testData, num_workers=NUM_WORKERS, batch_size=TEST_BATCH_SIZE, shuffle=False)
    return trainLoader, validLoader, testLoader


def mockfish_train(trainLoader, validLoader):
    mockfish = Mockfish(6, 64)
    mockfish.to(DEVICE)
    optimizer = Adam(mockfish.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    minValidLoss = np.inf
    bailCounter = 0

    trainLosses = []
    validLosses = []
    for x in range(EPOCHS):
        current_epoch = x+1
        print(f"Epoch {x+1}...")
        trainLoss = 0.0
        mockfish.train()
        for data, label,_,_ in tqdm(trainLoader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            target = mockfish(data)
            loss = criterion(target,label)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()
        
        validLoss = 0.0
        mockfish.eval()     
        with torch.no_grad():
            num_correct = 0.0
            num_samples = 0.0
            print("Validating...") 
            for data, label,_,_ in tqdm(validLoader):
                data, label = data.to(DEVICE), label.to(DEVICE)
                target = mockfish(data)
                _,predictions = torch.max(target.data, 1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)
                loss = criterion(target,label)
                validLoss += loss.item()

        accuracy = num_correct / num_samples

        avgTrainLoss = trainLoss / len(trainLoader)
        avgValidLoss = validLoss / len(validLoader)

        trainLosses.append(avgTrainLoss)
        validLosses.append(avgValidLoss)
    
        print(f'Epoch {x+1} \t\t Training loss: {avgTrainLoss} \t\t \
                Validation loss: {avgValidLoss} \t\t Validation accuracy: {100* accuracy:.2}%')

        if avgValidLoss > minValidLoss:
            # has validLoss failed to decrease twice in a row?
            bailCounter += 1
            if bailCounter >1:
                break
        else:
            bailCounter = 0
            minValidLoss = avgValidLoss
            torch.save(mockfish.state_dict(), MODELS_DIR + f"2fc_{current_epoch}e_{LEARNING_RATE}lr.pth")
            losses = pd.DataFrame({"trainLosses": trainLosses, "validLosses": validLosses})
            losses.to_csv(MODELS_DIR + f"fullmodel_{current_epoch}e_{LEARNING_RATE}lr_losses.csv")

    # SAVING WRONG MODEL - fix later

    return mockfish


def mockfish_test(model, testLoader):

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for data, label,_,_ in tqdm(testLoader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
    accuracy = float(num_correct) / float(num_samples)
    print(f'Got {num_correct} / {num_samples} with accuracy {(100 * num_correct/num_samples) :.2f}')
    model.train()
    return accuracy



if __name__=="__main__":
    print(f"Using device: {DEVICE}")
    print(f"num_workers: {NUM_WORKERS}")
    trainLoader, validLoader, testLoader = create_dataloaders()
    #model = mockfish_train(trainLoader, validLoader)
    model = Mockfish(6, 64).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR + "2fc_4e_0.001lr.pth"))
    accuracy = mockfish_test(model, testLoader)
    print(f"Test Accuracy: {accuracy}")