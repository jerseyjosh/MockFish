import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
from config import *
from functions import *
from argparse import ArgumentParser
from custom_torch_objects import Mockfish

# Add argparse for testing specific networks
parser = ArgumentParser()
parser.add_argument("-p", "--piece", help="choose the piece to test accuracy on.")
args = parser.parse_args()

PIECE = args.piece
if PIECE is not None:
    PIECE = PIECE.lower()
assert PIECE in ['None', 'p', 'b', 'n', 'r', 'q', 'k'], f"Expected one of [None, 'p', 'b', 'n', 'r', 'q', 'k'], got '{PIECE}'"

def get_model_path(dir, piece):
    pattern = f"*_{piece}_*.pth"
    for f in os.listdir(dir):
        if pattern.match(f):
            return dir + f

def test_model(model, testLoader):
    num_correct = 0.0
    num_samples = 0.0
    class_accuracies = [0 for _ in range(64)]
    model.eval()
    with torch.no_grad():
        for b, (data, from_square, to_square, _) in enumerate(tqdm(testLoader)):
            # set label by whether training piece selector or piece network
            label = from_square if PIECE is None else to_square
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



if __name__ == "__main__":
    model = Mockfish(6, 64).to(DEVICE)
    model_path = get_model_path(MODELS_DIR, PIECE)
    print("model path")
    model.load_state_dict(torch.load(model_path))
    trainLoader, validLoader, testLoader = create_dataloaders(piece=PIECE)
    accuracy, class_accuracies = test_model(model, testLoader)