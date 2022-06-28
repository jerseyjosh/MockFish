from tqdm import tqdm
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from custom_torch_objects import *
from config import *
from argparse import ArgumentParser
from functions import *

# Add argparse for training piece selector vs piece networks
parser = ArgumentParser()
parser.add_argument("-p", "--piece", help="choose the piece to train network on, if none is given then piece selector network is trained.")
args = parser.parse_args()

PIECE = args.piece
if PIECE is not None:
    PIECE = PIECE.lower()
assert PIECE in [None, 'p', 'b', 'n', 'r', 'q', 'k', 'all'], f"Expected one of [None, 'p', 'b', 'n', 'r', 'q', 'k', 'all'], got '{PIECE}'"

# TODO: fix saving most accurate model instead of most recent
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

        print(f"Epoch {current_epoch}...")
        trainLoss = 0.0
        mockfish.train()
        for batch,(data, from_square, to_square,_) in enumerate(tqdm(trainLoader)): 
            current_batch = batch+1

            running_iterations += 1
            # set label by whether training piece selector or piece network
            label = from_square if PIECE is None else to_square
            # save (data, label) to device
            data, label = data.to(DEVICE), label.to(DEVICE)

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
                    best_model = copy.deepcopy(mockfish)
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
        break

    # save best model
    if PIECE is not None:
        torch.save(best_model.state_dict(), MODELS_DIR + modelname + f"_{PIECE}_{current_epoch}e_{current_batch}b_{LEARNING_RATE}lr.pth")
    else:
        torch.save(best_model.state_dict(), MODELS_DIR + modelname + f"_selector_{current_epoch}e_{current_batch}b_{LEARNING_RATE}lr.pth")


    losses = pd.DataFrame({"iterations": iterations, "trainLosses": trainLosses, "validLosses": validLosses, "randomLosses": randomLosses})
    accuracies = pd.DataFrame({"iterations": iterations, "validAccuracies": validAccuracies, "randomAccuracies": randomAccuracies})

    if PIECE is not None:
        losses.to_csv(RESULTS_DIR + modelname + f"_{PIECE}_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr_losses.csv", index=False)
        accuracies.to_csv(RESULTS_DIR + modelname + f"_{PIECE}_{current_epoch}e_{LEARNING_RATE}lr_accuracies.csv", index=False)
    else:
        losses.to_csv(RESULTS_DIR + modelname + f"_pieceselector_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr_losses.csv", index=False)
        accuracies.to_csv(RESULTS_DIR + modelname + f"_pieceselector_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr_accuracies.csv", index=False)
    return mockfish


if __name__=="__main__":
    print(f"Using device: {DEVICE}")
    print(f"num_workers: {NUM_WORKERS}")
    if PIECE == 'all':
        print("Training all networks...")
        for p in [None, 'p', 'b', 'n', 'r', 'q', 'k']:
            trainLoader, validLoader, testLoader = create_dataloaders(piece=p)
            model = mockfish_train(trainLoader, validLoader, Mockfish, "mockfish")
    else:
        trainLoader, validLoader, testLoader = create_dataloaders(piece=PIECE)
        model = mockfish_train(trainLoader, validLoader, Mockfish, "mockfish")
