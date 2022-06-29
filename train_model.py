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

# create dataloaders for specific pieces
# PROBLEM IS IN DATALOADERS OR DATASET
def create_dataloaders(target_piece):
    print(f"Training {target_piece} network...")
    print("Loading dataset...")
    t0=time.time()
    chessData = ChessDataset(DATA_DIR, DF_PATH, target_piece=target_piece)
    t1=time.time()
    print(f"Took {t1-t0:.2f}s")

    print("Generating train/test/validation split...")
    train_size = int(TRAIN_SIZE * len(chessData))
    valid_size = int(VALID_SIZE * len(chessData))
    test_size = len(chessData) - train_size - valid_size

    trainData, validData, testData = torch.utils.data.random_split(chessData, [train_size, valid_size, test_size], 
                                                                   generator=torch.Generator().manual_seed(69))

    trainLoader = DataLoader(trainData, num_workers=NUM_WORKERS, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validLoader = DataLoader(validData, num_workers=NUM_WORKERS, batch_size=VALID_BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(testData, num_workers=NUM_WORKERS, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return trainLoader, validLoader, testLoader



def mockfish_train(trainLoader, validLoader, ModelClass, save_dir, target_piece='selector'):

    model = ModelClass(6, 64).to(DEVICE)
    print(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
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
            VALIDATION_EPOCHS = len(trainLoader) // 5
            if (batch+1) % VALIDATION_EPOCHS == 0:
                validLoss = 0.0
                vrandomLoss = 0.0
                model.eval()     
                with torch.no_grad():
                    vnum_correct = 0.0
                    vrandom_num_correct = 0.0
                    vnum_samples = 0.0
                    print("Validating...") 
                    for vdata, vfrom_square, vto_square,_ in tqdm(validLoader):
                        vlabel = vfrom_square if target_piece=='selector' else vto_square
                        vdata, vlabel = vdata.to(DEVICE), vlabel.to(DEVICE)
                        vtarget = model(vdata)

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
                    best_model = copy.deepcopy(model)
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
    torch.save(best_model.state_dict(), save_dir + ModelClass._get_name() + f"_{target_piece}_{current_epoch}e_{current_batch}b_{LEARNING_RATE}lr.pth")

    # save losses and accuracies
    losses = pd.DataFrame({"iterations": iterations, "trainLosses": trainLosses, "validLosses": validLosses, "randomLosses": randomLosses})
    accuracies = pd.DataFrame({"iterations": iterations, "validAccuracies": validAccuracies, "randomAccuracies": randomAccuracies})
    losses.to_csv(RESULTS_DIR + ModelClass._get_name() + f"_{target_piece}_{current_epoch}e_{batch+1}b_{LEARNING_RATE}lr_losses.csv", index=False)
    accuracies.to_csv(RESULTS_DIR + ModelClass._get_name() + f"_{target_piece}_{current_epoch}e_{LEARNING_RATE}lr_accuracies.csv", index=False)

if __name__=="__main__":
    print(f"Using device: {DEVICE}")
    print(f"num_workers: {NUM_WORKERS}") 

    if INPUT == 'all':
        print("Training all networks...")
        for p in ['selector', 'p', 'b', 'n', 'r', 'q', 'k']:
            trainLoader, validLoader, testLoader = create_dataloaders(target_piece=p)
            mockfish_train(trainLoader, validLoader, MockfishBaseline, save_dir=MODELS_DIR, target_piece=p)
    else:
        trainLoader, validLoader, testLoader = create_dataloaders(target_piece=INPUT)
        mockfish_train(trainLoader, validLoader, MockfishBaseline, save_dir = MODELS_DIR, target_piece=INPUT)
