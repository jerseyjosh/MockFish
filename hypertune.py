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
from train_model import mockfish_train
import optuna

def objective(trial):
    
    