import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import chess
from config import *
from functions import *
from custom_torch_objects import *



def combined_predict_move(fen, ps_path, p_path, b_path, n_path, r_path, q_path, k_path, num_moves=10, white_turn=True):

    best_moves = []

    board_state = torch.from_numpy(fen_to_board(fen=fen, piece_values=PIECE_VALUES, white_turn=white_turn))[None, :]
    board = chess.Board(fen=fen)
    board.turn = white_turn

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

    from_square_scores = F.softmax(ps_network(board_state), dim=1).cpu().detach().numpy()
    p_to_scores = F.softmax(p_network(board_state), dim=1).cpu().detach().numpy()
    b_to_scores = F.softmax(b_network(board_state), dim=1).cpu().detach().numpy()
    n_to_scores = F.softmax(n_network(board_state), dim=1).cpu().detach().numpy()
    r_to_scores = F.softmax(r_network(board_state), dim=1).cpu().detach().numpy()
    q_to_scores = F.softmax(q_network(board_state), dim=1).cpu().detach().numpy()
    k_to_scores = F.softmax(k_network(board_state), dim=1).cpu().detach().numpy()

    p_move_scores = from_square_scores.T @ p_to_scores
    b_move_scores = from_square_scores.T @ b_to_scores
    n_move_scores = from_square_scores.T @ n_to_scores
    r_move_scores = from_square_scores.T @ r_to_scores
    q_move_scores = from_square_scores.T @ q_to_scores
    k_move_scores = from_square_scores.T @ k_to_scores

    total_move_scores = p_move_scores + b_move_scores + n_move_scores + r_move_scores + q_move_scores + k_move_scores

    move_preference = np.squeeze(
        np.dstack(
            np.unravel_index(
                np.argsort(-total_move_scores.ravel()), total_move_scores.shape)))

    move_preference = np.array(move_preference)

    if not white_turn:
        move_preference = 63-move_preference

    for move in move_preference:
        if board.is_legal(chess.Move(move[0], move[1])):
            best_moves.append(move)
            if len(best_moves) > num_moves-1:
                return np.squeeze(best_moves)


if __name__=='__main__':
    with torch.no_grad():
        board = chess.Board()
        fen = board.fen()
        ps_path = get_model_path(MODELS_DIR, 'selector')
        p_path = get_model_path(MODELS_DIR, 'p')
        b_path = get_model_path(MODELS_DIR, 'b')
        n_path = get_model_path(MODELS_DIR, 'n')
        r_path = get_model_path(MODELS_DIR, 'r')
        q_path = get_model_path(MODELS_DIR, 'q')
        k_path = get_model_path(MODELS_DIR, 'k')

        print(combined_predict_move(
            fen, ps_path, p_path, b_path, n_path, r_path, q_path, k_path, num_moves=1, white_turn=board.turn))