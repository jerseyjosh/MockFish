from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from custom_torch_objects import *
from config import *
from argparse import ArgumentParser
from functions import *
from train_model import mockfish_train
import optuna

def objective(trial):
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 7),
        "hidden_size": trial.suggest_int("hidden_size", 32, 1024, 32),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.7),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
    }

    trainLoader = create_dataloaders(dir=DATA_DIR, path="training_2000elo.pickle", target_piece='selector')
    validLoader = create_dataloaders(dir=DATA_DIR, path='validation_2000elo.pickle', target_piece='selector')

    _,accuracy = mockfish_train(
        trainLoader, 
        validLoader, 
        Mockfish, 
        model_save_dir=None, 
        results_save_dir=None, 
        params=params,
        target_piece='selector',
        save_model=False
    )

    return accuracy


if __name__=="__main__":
    print("Creating study...")
    study = optuna.create_study(direction="maximize")
    print("Optimizing study...")
    study.optimize(objective, n_trials=20)
    print("Saving study...")
    pickle.dump(study, RESULTS_DIR + "hypertuning/study.pickle")
    best_trial = study.best_trial  

    print("Best Trial: ")
    print(best_trial.values)
    print(best_trial.params)

    trainLoader = create_dataloaders(dir=DATA_DIR, path="training_2000elo.pickle", target_piece='selector')
    validLoader = create_dataloaders(dir=DATA_DIR, path='validation_2000elo.pickle', target_piece='selector')

    _,best_accuracy = mockfish_train(
        trainLoader, 
        validLoader, 
        Mockfish, 
        model_save_dir=MODELS_DIR + "hypertuning/", 
        results_save_dir=RESULTS_DIR + "hypertuning/", 
        params=best_trial.params,
        target_piece='selector',
        save_model=True
    )

    print("Best Accuracy:")
    print(best_accuracy)