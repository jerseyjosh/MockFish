import torch
import torch.nn as nn

# Training Params
ELO_LOWER_LIMIT = 2000
PIECE_VALUES = [1, 1, 1, 1, 1, 1]
TRAIN_SIZE, TRAIN_BATCH_SIZE = 0.8, 250
VALID_SIZE, VALID_BATCH_SIZE = 0.1, 250
TEST_SIZE, TEST_BATCH_SIZE = 0.1, 250
NUM_WORKERS = 0 # Apple Silicon crashes with num_workers > 0
EPOCHS = 10
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
LEARNING_RATE = 0.001 # ADAM adapts automatically, tuning can be ignored

# Paths
IMAGE_DIR = "./images/"
DATA_DIR = "./data/"
DATA_PATH = "2016_CvC.csv"
DF_PATH = f"data_{ELO_LOWER_LIMIT}elo.pickle"
TRAINING_PATH = f"training_{ELO_LOWER_LIMIT}elo.pickle"
VALIDATION_PATH = f"validation_{ELO_LOWER_LIMIT}elo.pickle"
TESTING_PATH = f"testing_{ELO_LOWER_LIMIT}elo.pickle"
MODELS_DIR = "./models/"
TEMP_MODELS_DIR = "./temp_models/"
MODEL_PATH = f"mockfish_{EPOCHS}epochs_{LEARNING_RATE}lr_{PIECE_VALUES}.pickle"
RESULTS_DIR = "./results/"
