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
parser.add_argument("piece", 
                    help="choose the piece to train network on.")
args = parser.parse_args()

INPUT = args.piece
if INPUT is not None:
    INPUT = INPUT.lower()
assert INPUT in ['selector', 'p', 'b', 'n', 'r', 'q', 'k', 'all'], f"Expected one of ['selector', 'p', 'b', 'n', 'r', 'q', 'k', 'all'], got '{INPUT}'"


def mockfish_train(trainLoader,
                    validLoader, 
                    ModelClass, 
                    model_save_dir, 
                    results_save_dir, 
                    target_piece='selector',
                    learning_rate = LEARNING_RATE):

    model = ModelClass().to(DEVICE)
    print(model)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainLosses = []
    validLosses = []
    iterations = []
    iterations_proportions = []
    running_iterations = 0
    validAccuracies = []

    minValidLoss = np.inf
    bailCounter = 0

    for x in range(EPOCHS):
        current_epoch = x+1

        print(f"Epoch {current_epoch}...")
        trainLoss = 0.0
        model.train()
        for batch,(data, from_square, to_square,_) in enumerate(tqdm(trainLoader)): 

            current_batch = batch+1
            running_iterations += 1

            # set label by whether training piece selector or piece network
            label = from_square if target_piece=='selector' else to_square
            # save (data, label) to device
            data, label = data.to(DEVICE), label.to(DEVICE)

            # training
            optimizer.zero_grad()
            target = model(data)
            loss = criterion(target,label)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()

            # Validate 5 times per epoch
            NUM_BATCHES_BEFORE_VALIDATION = len(trainLoader) // 5
            if (batch+1) % NUM_BATCHES_BEFORE_VALIDATION == 0:
                validLoss = 0.0
                model.eval()     
                with torch.no_grad():
                    vnum_correct = 0.0
                    vnum_samples = 0.0
                    print("Validating...") 
                    for vdata, vfrom_square, vto_square,_ in tqdm(validLoader):
                        vlabel = vfrom_square if target_piece=='selector' else vto_square
                        vdata, vlabel = vdata.to(DEVICE), vlabel.to(DEVICE)
                        vtarget = model(vdata)

                        _,vpredictions = torch.max(vtarget.data, 1)

                        vnum_correct += (vpredictions == vlabel).sum()
                        vnum_samples += vpredictions.size(0)

                        vloss = criterion(vtarget,vlabel)

                        validLoss += vloss.item()

                avgTrainLoss = trainLoss / (batch+1)
                avgValidLoss = validLoss / len(validLoader)

                vaccuracy = float(vnum_correct) / float(vnum_samples)

                iterations.append(running_iterations)
                iterations_proportions.append(running_iterations / len(trainLoader))
                trainLosses.append(avgTrainLoss)
                validLosses.append(avgValidLoss)
                validAccuracies.append(vaccuracy)

                print(f"Iteration {batch+1}")
                print(f"Training loss: {avgTrainLoss} \t\t Validation loss: {avgValidLoss} \t\t Validation accuracy: {100 * vaccuracy :.2}%")

                if avgValidLoss < minValidLoss:
                    best_model = copy.deepcopy(model)
                    minValidLoss = avgValidLoss
                    print("Bail counter reset.")
                    bailCounter = 0

                else:
                    bailCounter += 1
                    print(f"Validation loss increased, bail counter: {bailCounter}")
                    if bailCounter > 4:
                        break
                    
        # for/else block to break outer loop when vloss stops decreasing
        else:
            continue
        break

    # save best model
    torch.save(best_model.state_dict(), model_save_dir + model._get_name() + f"{target_piece}_{current_epoch}e_{current_batch}b_{LEARNING_RATE}lr.pth")

    # save losses and accuracies
    losses = pd.DataFrame(
        {"iterations": iterations, 
        "iterations_proportions": iterations_proportions, 
        "trainLosses": trainLosses, 
        "validLosses": validLosses})
    accuracies = pd.DataFrame(
        {"iterations": iterations, 
        "iterations_proportions": iterations_proportions, 
        "validAccuracies": validAccuracies})
    losses.to_csv(results_save_dir + model._get_name() + f"_{target_piece}_{LEARNING_RATE}lr_losses.csv", index=False)
    accuracies.to_csv(results_save_dir + model._get_name() + f"_{target_piece}_{LEARNING_RATE}lr_accuracies.csv", index=False)

if __name__=="__main__":

    print(f"Using device: {DEVICE}")
    print(f"num_workers: {NUM_WORKERS}") 

    if INPUT == 'all':
        print("Training all networks...")
        for p in ['selector', 'p', 'b', 'n', 'r', 'q', 'k']:
            trainLoader = create_dataloaders(target_piece=p, dir=DATA_DIR, path="training_2000elo.pickle")
            validLoader = create_dataloaders(target_piece=p, dir=DATA_DIR, path='validation_2000elo.pickle')
            mockfish_train(
                trainLoader, validLoader, Mockfish, 
                model_save_dir=MODELS_DIR, results_save_dir=RESULTS_DIR+'training/', target_piece=p)
    else:
        trainLoader = create_dataloaders(target_piece=INPUT, dir=DATA_DIR, path="training_2000elo.pickle")
        validLoader = create_dataloaders(target_piece=INPUT, dir=DATA_DIR, path="validation_2000elo.pickle")
        mockfish_train(
            trainLoader, validLoader, Mockfish, 
            model_save_dir=MODELS_DIR, results_save_dir=RESULTS_DIR+'training/', target_piece=INPUT)
