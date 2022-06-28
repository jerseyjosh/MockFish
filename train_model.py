import pickle
from tqdm import tqdm
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
from custom_torch_objects import *
from config import *
from argparse import ArgumentParser

# Add argparse for training piece selector vs piece networks
parser = ArgumentParser()
parser.add_argument("-p", "--piece", help="choose the piece to train network on, if none is given then piece selector network is trained.")
args = parser.parse_args()

PIECE = args.piece.lower()
assert len(PIECE)==1, f"Expected length 1, got {len(PIECE)}"
assert PIECE.lower() in ['p', 'b', 'n', 'r', 'q', 'k'], f"Expected one of ['p', 'b', 'n', 'r', 'q', 'k'], got '{PIECE}'"

def create_dataloaders(piece=PIECE):
    if piece is not None:
        print(f"Training '{PIECE}' network...")
    else:
        print("Training piece selector network...")
    print("Loading dataset...")
    t0=time.time()
    chessData = ChessDataset(DATA_DIR, DF_PATH, piece=piece)
    t1=time.time()
    print(f"Took {t1-t0:.2f}s")

    print("Generating train/test/validation split...")
    train_size = int(TRAIN_SIZE * len(chessData))
    valid_size = int(VALID_SIZE * len(chessData))
    test_size = len(chessData) - train_size - valid_size

    trainData, validData, testData = torch.utils.data.random_split(chessData, [train_size, valid_size, test_size], 
                                                                   generator=torch.Generator().manual_seed(1))

    trainLoader = DataLoader(trainData, num_workers=NUM_WORKERS, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validLoader = DataLoader(validData, num_workers=NUM_WORKERS, batch_size=VALID_BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(testData, num_workers=NUM_WORKERS, batch_size=TEST_BATCH_SIZE, shuffle=False)
    return trainLoader, validLoader, testLoader


def mockfish_train(trainLoader, validLoader, Model, modelname: str):

    mockfish = Model(6, 64).to(DEVICE)
    optimizer = Adam(mockfish.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    trainLosses = []
    validLosses = []
    randomLosses = []
    iterations = []
    running_iterations = 0
    
    validAccuracies = []
    randomAccuracies = []

    minValidLoss = np.inf
    bailCounter = 0

    for x in range(EPOCHS):
        current_epoch = x+1
        print(f"Epoch {x+1}...")
        trainLoss = 0.0
        mockfish.train()
        for batch,(data, from_square, to_square,_) in enumerate(tqdm(trainLoader)): 
            # TODO: THIS NEEDS TO BE MADE DYNAMIC
            # set label by whether training piece selector or piece network
            label = from_square if PIECE is None else to_square
            # save (data, label) to device
            data, label = data.to(DEVICE), label.to(DEVICE)
            running_iterations += batch+1
            # training
            optimizer.zero_grad()
            target = mockfish(data)
            loss = criterion(target,label)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()

            # Validate 5 times per epoch
            VALIDATION_EPOCHS = len(trainLoader) // 5
            if (batch+1) % VALIDATION_EPOCHS == 0:
                validLoss = 0.0
                vrandomLoss = 0.0
                mockfish.eval()     
                with torch.no_grad():
                    vnum_correct = 0.0
                    vrandom_num_correct = 0.0
                    vnum_samples = 0.0
                    print("Validating...") 
                    for vdata, vfrom_square, vto_square,_ in tqdm(validLoader):
                        vlabel = vfrom_square if PIECE is None else vto_square
                        vdata, vlabel = vdata.to(DEVICE), vlabel.to(DEVICE)
                        vtarget = mockfish(vdata)

                        _,vpredictions = torch.max(vtarget.data, 1)
                        vrandom_predictions = torch.randint(64, [len(vdata)]).to(DEVICE)
                        vrandom_probs = torch.Tensor(len(vdata) * [[1/64] * 64]).to(DEVICE)

                        vnum_correct += (vpredictions == vlabel).sum()
                        vrandom_num_correct += (vrandom_predictions == vlabel).sum()
                        vnum_samples += vpredictions.size(0)

                        vloss = criterion(vtarget,vlabel)
                        vrandom_loss = criterion(vrandom_probs, vlabel)

                        validLoss += vloss.item()
                        vrandomLoss += vrandom_loss.item()

                avgTrainLoss = trainLoss / (batch+1)
                avgValidLoss = validLoss / len(validLoader)
                avgRandomLoss = vrandomLoss / len(validLoader)

                vaccuracy = float(vnum_correct) / float(vnum_samples)
                vrandom_accuracy = float(vrandom_num_correct) / float(vnum_samples)

                iterations.append(running_iterations)
                trainLosses.append(avgTrainLoss)
                validLosses.append(avgValidLoss)
                randomLosses.append(avgRandomLoss)

                validAccuracies.append(vaccuracy)
                randomAccuracies.append(vrandom_accuracy)

                print(f"Iteration {batch+1}")
                print(f"Training loss: {avgTrainLoss} \t\t Validation loss: {avgValidLoss} \t\t Random loss: {avgRandomLoss}")
                print(f"Validation accuracy: {100 * vaccuracy :.2}% \t\t Random Accuracy: {100 * vrandom_accuracy :.2f}%")

                if avgValidLoss < minValidLoss:
                    minValidLoss = avgValidLoss
                    print("Bail counter reset.")
                    bailCounter = 0

                else:
                    bailCounter += 1
                    print(f"Validation loss increased, bail counter: {bailCounter}")
                    if bailCounter > 2:
                        break
                    
        # for/else block to break outer loop when vloss stops decreasing
        else:
            continue
        if PIECE is not None:
            torch.save(mockfish.state_dict(), MODELS_DIR + modelname + f"_{PIECE}_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr.pth")
        else:
            torch.save(mockfish.state_dict(), MODELS_DIR + modelname + f"_pieceselector_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr.pth")
        break

    losses = pd.DataFrame({"iterations": iterations, "trainLosses": trainLosses, "validLosses": validLosses, "randomLosses": randomLosses})
    accuracies = pd.DataFrame({"iterations": iterations, "validAccuracies": validAccuracies, "randomAccuracies": randomAccuracies})

    if PIECE is not None:
        losses.to_csv(MODELS_DIR + modelname + f"_{PIECE}_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr_losses.csv", index=False)
        accuracies.to_csv(MODELS_DIR + modelname + f"_{PIECE}_{current_epoch}e_{LEARNING_RATE}lr_accuracies.csv", index=False)
    else:
        losses.to_csv(MODELS_DIR + modelname + f"_pieceselector_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr_losses.csv", index=False)
        accuracies.to_csv(MODELS_DIR + modelname + f"_pieceselector_{current_epoch}e_{LEARNING_RATE}lr_accuracies.csv", index=False)
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
            ###
            # TODO: IMPLEMENT MULTICLASS ACCURACY, maybe visualisation
            ###
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
    trainLoader, validLoader, testLoader = create_dataloaders(piece='p')
    model = mockfish_train(trainLoader, validLoader, Mockfish, "mockfish")
    #model = Mockfish(6, 64).to(DEVICE)
    #model.load_state_dict(torch.load(MODELS_DIR + "fullmodel_1e_0.0015lr.pth"))
    accuracy, class_accuracies = mockfish_test(model, testLoader)
    print(f"Test Accuracy: {accuracy*100 :.2f}%")