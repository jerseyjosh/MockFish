import torch
import torch.nn as nn

ELO_LOWER_LIMIT = 2000
PIECE_VALUES = [1, 1, 1, 1, 1, 1]
DATA_DIR = "./data/"
DATA_PATH = "2016_CvC.csv"
DF_PATH = f"training_data_{ELO_LOWER_LIMIT}.pickle"

# Training Params
TRAIN_SIZE, TRAIN_BATCH_SIZE = 0.8, 64
VALID_SIZE, VALID_BATCH_SIZE = 0.1, 128
TEST_SIZE, TEST_BATCH_SIZE = 0.1, 128
NUM_WORKERS = 0
EPOCHS = 10
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
LEARNING_RATE = 0.001
MODELS_DIR = "./models/"
MODEL_PATH = f"mockfish_{EPOCHS}epochs_{LEARNING_RATE}lr_{PIECE_VALUES}.pickle"