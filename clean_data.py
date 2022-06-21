import pandas as pd
import numpy as np
from config import *

chess_data = pd.read_csv(DATA_DIR + "2016_CvC.csv")
chess_data = chess_data[((chess_data["White Elo"] > ELO_LOWER_LIMIT)&(chess_data["Black Elo"]> ELO_LOWER_LIMIT))]

if __name__=="__main__":
    print(chess_data.head())