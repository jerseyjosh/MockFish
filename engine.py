from custom_torch_objects import Mockfish
import torch
import chess
from config import *
from functions import *

class Engine():
    def __init__(self, selector_path, p_path, b_path, n_path, r_path, q_path, k_path):
        self.selector = Mockfish(6, 64)
        self.selector.load_state_dict(torch.load(selector_path))
        self.selector.eval()

        self.pawn = Mockfish(6, 64)
        self.pawn.load_state_dict(torch.load(p_path))
        self.pawn.eval()

        self.bishop = Mockfish(6, 64)
        self.bishop.load_state_dict(torch.load(b_path))
        self.bishop.eval()

        self.knight = Mockfish(6, 64)
        self.knight.load_state_dict(torch.load(n_path))
        self.knight.eval()

        self.rook = Mockfish(6, 64)
        self.rook.load_state_dict(torch.load(r_path))
        self.rook.eval()

        self.queen = Mockfish(6, 64)
        self.queen.load_state_dict(torch.load(q_path))
        self.queen.eval()

        self.king = Mockfish(6, 64)
        self.king.load_state_dict(torch.load(k_path))
        self.king.eval()

if __name__=="__main__":
    selector_path = get_model_path(MODELS_DIR, 'selector')
    print(selector_path)
    p_path = get_model_path(MODELS_DIR, 'p')
    b_path = get_model_path(MODELS_DIR, 'b')
    n_path = get_model_path(MODELS_DIR, 'n')
    r_path = get_model_path(MODELS_DIR, 'r')
    q_path = get_model_path(MODELS_DIR, 'q')
    k_path = get_model_path(MODELS_DIR, 'k')

    print("Initialising JoshFish 1.0")
    engine = Engine(selector_path, p_path, b_path, n_path, r_path, q_path, k_path)

    board = chess.Board()

    while not board.is_game_over():
        # my move
        print(board)
        move = input("Your move: ")
        board.push_san(move)
        fen = board.board_fen()