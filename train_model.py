import pickle
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

    mockfish = Mockfish(6, 64).to(DEVICE)
    optimizer = Adam(mockfish.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    minValidLoss = np.inf
    bailCounter = 0

    trainLosses = []
    validLosses = []
    randomLosses = []
    iterations = []
    
    validAccuracies = []
    randomAccuracies = []

    for x in range(EPOCHS):
        current_epoch = x+1
        print(f"Epoch {x+1}...")
        trainLoss = 0.0
        mockfish.train()
        for batch,(data, label,_,_) in enumerate(tqdm(trainLoader)):
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            target = mockfish(data)
            loss = criterion(target,label)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()

            # Validate
            if (batch+1) % 1000 == 0:
                validLoss = 0.0
                randomLoss = 0.0
                mockfish.eval()     
                with torch.no_grad():
                    num_correct = 0.0
                    random_num_correct = 0.0
                    num_samples = 0.0
                    print("Validating...") 
                    for data, label,_,_ in tqdm(validLoader):
                        data, label = data.to(DEVICE), label.to(DEVICE)
                        target = mockfish(data)
                        _,predictions = torch.max(target.data, 1)

                        random_predictions = torch.randint(64, [VALID_BATCH_SIZE]).to(DEVICE)
                        random_probs = torch.Tensor(250 * [[1/64] * 64]).to(DEVICE)

                        num_correct += (predictions == label).sum()
                        random_num_correct += (random_predictions == label).sum()
                        num_samples += predictions.size(0)

                        loss = criterion(target,label)
                        random_loss = criterion(random_probs, label)
                        validLoss += loss.item()
                        randomLoss += random_loss.item()

                avgTrainLoss = trainLoss / ((batch+1) * TRAIN_BATCH_SIZE)
                avgValidLoss = validLoss / len(validLoader)
                avgRandomLoss = randomLoss / len(validLoader)

                val_accuracy = num_correct / num_samples
                random_accuracy = random_num_correct / num_samples

                iterations.append((batch+1) * TRAIN_BATCH_SIZE)
                trainLosses.append(avgTrainLoss)
                validLosses.append(avgValidLoss)
                randomLosses.append(avgRandomLoss)

                validAccuracies.append(val_accuracy)
                randomAccuracies.append(random_accuracy)
    
                print(f'Iteration {(batch+1) * TRAIN_BATCH_SIZE} \t\t Training loss: {avgTrainLoss} \t\t \
                        Validation loss: {avgValidLoss} \t\t Validation accuracy: {100 * accuracy:.2}%')

        torch.save(mockfish.state_dict(), MODELS_DIR + f"fullmodel_{current_epoch}e_{LEARNING_RATE}lr.pth")

        if avgValidLoss < minValidLoss:
            minValidLoss = avgValidLoss
            bailCounter = 0

        else:
            bailCounter += 1
            if bailCounter > 2:
                break

    losses = pd.DataFrame(index=np.arange(0, len(trainLosses)*TRAIN_BATCH_SIZE+1), \
                                        data={"trainLosses": trainLosses, 
                                                "validLosses": validLosses,
                                                "randomLosses": randomLosses})
    losses.to_csv(MODELS_DIR + f"fullmodel_{current_epoch}e_{LEARNING_RATE}lr_losses.csv", index=True)

    accuracies = pd.DataFrame({"validAccuracies": validAccuracies, "randomAccuracies": {randomAccuracies}})
    accuracies.to_csv(MODELS_DIR + f"fullmodel_{current_epoch}e_{LEARNING_RATE}lr_accuracies.csv", index=False)

    return mockfish


def mockfish_test(model, testLoader):
    num_correct = 0.0
    num_samples = 0.0
    class_accuracies = [0 for _ in range(64)]
    model.eval()
    with torch.no_grad():
        for b, (data, label,_,_) in enumerate(tqdm(testLoader)):
            data, label = data.to(DEVICE), label.to(DEVICE)
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
            #for i in range(64):
                 #class_accuracies[i] += ((predictions == label) * (label == i)).float() / (max(label == i).sum(), 1)
            #class_accuracies /= (b+1)
    accuracy = num_correct/num_samples

    print(f'Got {num_correct} / {num_samples} with accuracy {(100 * accuracy) :.2f}')
    model.train()

    return accuracy, class_accuracies


if __name__=="__main__":
    print(f"Using device: {DEVICE}")
    print(f"num_workers: {NUM_WORKERS}")
    trainLoader, validLoader, testLoader = create_dataloaders()
    model = mockfish_train(trainLoader, validLoader)
    model = Mockfish(6, 64).to(DEVICE)
    #model.load_state_dict(torch.load(MODELS_DIR + "fullmodel_6e_0.001lr.pth"))
    accuracy, class_accuracies = mockfish_test(model, testLoader)
    print(f"Test Accuracy: {accuracy}")