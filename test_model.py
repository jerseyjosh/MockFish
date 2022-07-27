import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import os
from config import *
from functions import *
from engine import Engine
from argparse import ArgumentParser
from custom_torch_objects import Mockfish

# Add argparse for testing specific networks
parser = ArgumentParser()
parser.add_argument("piece", 
                    help="choose the piece to test network on.")
args = parser.parse_args()

INPUT = args.piece
if INPUT is not None:
    INPUT = INPUT.lower()
assert INPUT in ['selector', 'p', 'b', 'n', 'r', 'q', 'k', 'all'], f"Expected one of ['selector', 'p', 'b', 'n', 'r', 'q', 'k', 'all'], got '{INPUT}'"

def get_model_path(dir, piece):
    pattern = f"puzzle_{piece}_"
    for f in os.listdir(dir):
        if re.search(pattern, f):
            return dir + f

def test_model(testLoader, ModelClass, target_piece='selector'):

    model = ModelClass().to(DEVICE)
    print(model)
    model_path = get_model_path(MODELS_DIR, target_piece)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    num_correct = 0.0
    num_samples = 0.0

    # initialize empty confusion matrix
    cm = torch.zeros((64, 64))

    with torch.no_grad():
        for b, (data, from_square, to_square, _) in enumerate(tqdm(testLoader)):
            # set label by whether training piece selector or piece network
            label = from_square if target_piece=='selector' else to_square
            data, label = data.to(DEVICE), label.to(DEVICE)
            scores = model(data)
            _, predictions = scores.max(1)

            # data needs to be on cpu for confusion matrix
            predictions, label = predictions.to('cpu'), label.to('cpu')
            cm += confusion_matrix(label, predictions)

            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
        
    accuracy = float(num_correct)/float(num_samples)
    class_accuracy = cm.diag()/cm.sum(1)

    print(f'Got {num_correct} / {num_samples} with accuracy {(100 * accuracy) :.2f}')

    return accuracy, class_accuracy, cm



if __name__ == "__main__":
    if INPUT == 'all':
        accuracies = []
        class_accuracies = []
        confusion_matrices = []
        print("Testing all networks...")
        for p in ['selector', 'p', 'b', 'n', 'r', 'q', 'k']:
            testLoader = create_dataloaders(target_piece=p, path="puzzle_test.pickle")
            accuracy, class_accuracy, cm = test_model(testLoader, Mockfish, target_piece=p)
            accuracies.append(accuracy)
            class_accuracies.append(class_accuracy)
            confusion_matrices.append(cm)
        df = pd.DataFrame(
            {'network':['selector', 'p', 'b', 'n', 'r', 'q', 'k'], 
            'accuracy':accuracies,
            'class_accuracy': class_accuracies,
            'confusion_matrix': confusion_matrices})
        df.to_pickle(RESULTS_DIR + "puzzle_testing_results.pickle")
    else:
        testLoader = create_dataloaders(target_piece=INPUT, path="puzzle_test.pickle")
        test_model(testLoader, Mockfish, target_piece=INPUT)