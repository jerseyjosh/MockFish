import pandas as pd
import numpy as np
from tqdm import tqdm
from config import *
from functions import *

def clean_puzzles(root, path, train_path, valid_path, test_path, random_seed=69):

    # Load CSV, drop NaN, get rid of negative popularity puzzles
    print("Loading puzzles csv...")
    puzzles = pd.read_csv(root + path, on_bad_lines='skip', header=None)
    puzzles.columns = ['PuzzleId','FEN','Moves','Rating','RatingDeviation','Popularity','NbPlays','Themes','GameUrl', 'OpeningFamily']
    puzzles = puzzles[["FEN", "Moves", "Rating", "Popularity", "Themes"]].dropna()
    puzzles = puzzles[puzzles.Popularity > 0]
    # not filtering ELO to allow simple mate-in-one puzzles also

    # convert to board states
    print("Generating board states...")
    board_states = []
    from_squares = []
    to_squares = []
    pieces_moved = []
    for i, puzzle in tqdm(puzzles.iterrows()):

        board = chess.Board(fen=puzzle.FEN)
        white_turn=board.turn

        for move in puzzle.Moves.split():
            from_square = chess.parse_square(move[:2])
            to_square = chess.parse_square(move[2:4])
            piece_moved = board.piece_at(from_square)
            board_array = fen_to_board(board.board_fen(), piece_values=PIECE_VALUES, white_turn=white_turn)

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
            
    print(f"Generated {len(puzzles)} samples.")

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

        



if __name__ == '__main__':

    clean_puzzles(
        root=DATA_DIR, 
        path='lichess_db_puzzle.csv.bz2', 
        train_path='puzzle_training.pickle',
        valid_path='puzzle_validation.pickle',
        test_path='puzzle_test.pickle',
        random_seed=69)