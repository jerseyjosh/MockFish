import pickle
import pandas as pd
import numpy as np
import chess
from tqdm import tqdm
import time
from functions import *
from config import *

def load_data(root, path, train_path, valid_path, test_path, elo_limit, piece_values, random_seed=69):

    # load csv and filter Elo
    print("Loading csv...")
    data = pd.read_csv(root + path)
    data = data[((data["White Elo"] > elo_limit)&(data["Black Elo"] > elo_limit))]

    # generate SANs
    print("Parsing SANs...")
    moves = data.Moves
    parsed_sans = moves.apply(lambda x: parse_san(x))

    # generate LANs
    print("Parsing LANs...")
    parsed_lans = parsed_sans.apply(lambda x: san_to_lan(x))
    print("Saving parsed_lans.csv...")
    parsed_lans.to_csv(root + f"parsed_lans_{elo_limit}elo.zip", header=False, index=False)

    # convert to board states
    print("Generating board states...")
    board_states = []
    from_squares = []
    to_squares = []
    pieces_moved = []
    for game in tqdm(parsed_lans):

        board = chess.Board()
        white_turn=True
        for move in game.split():
            from_square = chess.parse_square(move[:2])
            to_square = chess.parse_square(move[2:4])
            piece_moved = board.piece_at(from_square)
            board_array = fen_to_board(board.board_fen(), piece_values=piece_values, white_turn=white_turn)
            ### FROM_SQUARES AND TO_SQUARES ARE NOW ALWAYS COUNTING AS IF FROM PERSPECTIVE OF WHITE

            board_states.append(board_array)
            if white_turn:
                from_squares.append(from_square)
                to_squares.append(to_square)
            elif not white_turn:
                from_squares.append(63-from_square)
                to_squares.append(63-to_square)
            pieces_moved.append(str(piece_moved))

            board.push_san(move)
            white_turn = not white_turn

    df = pd.DataFrame({"board_states": board_states, 
                        "from_squares": from_squares, 
                        "to_squares": to_squares,
                        "pieces_moved": pieces_moved})

    print("Splitting into train/test/validation")   
    train, valid, test = np.split(df.sample(frac=1, random_state=random_seed), [int(.8*len(df)), int(.9*len(df))])
    print("Saving training data...")
    train.to_pickle(root + train_path)
    print("Saving validation data...")
    valid.to_pickle(root + valid_path)
    print("Saving testing data...")
    test.to_pickle(root + test_path)

        
if __name__ == "__main__":
    t0=time.time()
    load_data(
        root=DATA_DIR, 
        path=DATA_PATH,
        train_path='training_2000elo.pickle', 
        valid_path='validation_2000elo.pickle',
        test_path='testing_2000elo.pickle',
        elo_limit=ELO_LOWER_LIMIT,
        piece_values=PIECE_VALUES)
    t1=time.time()
    print("Done.")
    print(f"Took {t1-t0:.2f}s")