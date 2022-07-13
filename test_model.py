import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
from config import *
from functions import *
from argparse import ArgumentParser
from custom_torch_objects import Mockfish
from train_model import create_dataloaders

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
    pattern = f"_{piece}_"
    for f in os.listdir(dir):
        if re.search(pattern, f):
            return dir + f

def test_model(testLoader, ModelClass, save_dir, target_piece='selector'):
    model = ModelClass().to(DEVICE)
    print(model)
    model_path = get_model_path(TEMP_MODELS_DIR, target_piece)
    model.load_state_dict(torch.load(model_path))
    num_correct = 0.0
    num_samples = 0.0
    class_accuracies = [0 for _ in range(64)]
    model.eval()
    with torch.no_grad():
        for b, (data, from_square, to_square, _) in enumerate(tqdm(testLoader)):
            # set label by whether training piece selector or piece network
            label = from_square if target_piece=='selector' else to_square
            data, label = data.to(DEVICE), label.to(DEVICE)
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
            ###
            # TODO: IMPLEMENT MULTICLASS ACCURACY, maybe visualisation
            ###
    accuracy = float(num_correct)/float(num_samples)

    print(f'Got {num_correct} / {num_samples} with accuracy {(100 * accuracy) :.2f}')

    return accuracy, class_accuracies



if __name__ == "__main__":
    if INPUT == 'all':
        accuracies = []
        print("Testing all networks...")
        for p in ['selector', 'p', 'b', 'n', 'r', 'q', 'k']:
            testLoader = create_dataloaders(target_piece=p, dataset='testing')
            accuracy,_ = test_model(testLoader, Mockfish, save_dir=RESULTS_DIR, target_piece=p)
            accuracies.append(accuracy)
        df = pd.DataFrame({'network':['selector', 'p', 'b', 'n', 'r', 'q', 'k'], 'accuracy':accuracies})
        df.to_csv(RESULTS_DIR + "testing_accuracies.csv", index=False)
    else:
        testLoader = create_dataloaders(target_piece=INPUT, dataset='testing')
        test_model(testLoader, Mockfish, save_dir=RESULTS_DIR, target_piece=INPUT)