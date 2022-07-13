import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import chess
from config import *
from functions import *
from custom_torch_objects import *


def predict_move(board_state, ps_path, p_path, b_path, n_path, r_path, q_path, k_path):

    ps_network = Mockfish()
    ps_network.load_state_dict(torch.load(ps_path))
    ps_network.eval()

    p_network = Mockfish()
    p_network.load_state_dict(torch.load(p_path))
    p_network.eval()

    b_network = Mockfish()
    b_network.load_state_dict(torch.load(b_path))
    b_network.eval()

    n_network = Mockfish()
    n_network.load_state_dict(torch.load(n_path))
    n_network.eval()

    r_network = Mockfish()
    r_network.load_state_dict(torch.load(r_path))
    r_network.eval()

    q_network = Mockfish()
    q_network.load_state_dict(torch.load(q_path))
    q_network.eval()

    k_network = Mockfish()
    k_network.load_state_dict(torch.load(k_path))
    k_network.eval()

    from_square_scores = F.softmax(ps_network(board_state), dim=1)
    p_to_scores = F.softmax(p_network(board_state), dim=1)
    b_to_scores = b_network(board_state)
    n_to_scores = n_network(board_state)
    r_to_scores = r_network(board_state)
    q_to_scores = q_network(board_state)
    k_to_scores = k_network(board_state)

    p_move_scores = from_square_scores.T @ p_to_scores
    v, i = torch.topk(p_move_scores.flatten(), 5)
    print(np.array(np.unravel_index(i.numpy(), p_move_scores.shape)).T)



if __name__=='__main__':

    board = chess.Board()
    fen = board.board_fen()
    board_state = torch.from_numpy(
        fen_to_board(fen, piece_values=PIECE_VALUES, white_turn=board.turn))[None,:]

    ps_path = get_model_path(MODELS_DIR, 'selector')
    p_path = get_model_path(MODELS_DIR, 'p')
    b_path = get_model_path(MODELS_DIR, 'b')
    n_path = get_model_path(MODELS_DIR, 'n')
    r_path = get_model_path(MODELS_DIR, 'r')
    q_path = get_model_path(MODELS_DIR, 'q')
    k_path = get_model_path(MODELS_DIR, 'k')

    print(board)

    predict_move(board_state, ps_path, p_path, b_path, n_path, r_path, q_path, k_path)