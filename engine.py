from custom_torch_objects import Mockfish
import torch.nn.functional as F
import torch
import chess
import chess.svg
from IPython.display import SVG, display
from config import *
from functions import *

class Engine():
    def __init__(self, selector_path, p_path, b_path, n_path, r_path, q_path, k_path):

        self.selector = Mockfish()
        self.selector.load_state_dict(torch.load(selector_path))
        self.selector.eval()

        self.pawn = Mockfish()
        self.pawn.load_state_dict(torch.load(p_path))
        self.pawn.eval()

        self.bishop = Mockfish()
        self.bishop.load_state_dict(torch.load(b_path))
        self.bishop.eval()

        self.knight = Mockfish()
        self.knight.load_state_dict(torch.load(n_path))
        self.knight.eval()

        self.rook = Mockfish()
        self.rook.load_state_dict(torch.load(r_path))
        self.rook.eval()

        self.queen = Mockfish()
        self.queen.load_state_dict(torch.load(q_path))
        self.queen.eval()

        self.king = Mockfish()
        self.king.load_state_dict(torch.load(k_path))
        self.king.eval()


    def play(self):
        board = chess.Board()
        while not board.is_game_over():
            display(chess.svg.board(board, size=500))
            #print(board)
            move_is_legal = False
            while not move_is_legal:
                move = input("Your move: ")
                try:
                    board.push_san(move)
                    move_is_legal = True
                except ValueError:
                    print("Illegal move, try again.")
            fen = board.fen()
            from_square, piece_moved = self.get_from_square(fen=fen, white_turn=board.turn)
            to_square = self.get_to_square(fen=fen, white_turn=board.turn, from_square=from_square, piece_moved=piece_moved)
            #print(chess.square_name(from_square), chess.square_name(to_square))
            board.push(chess.Move(from_square, to_square))
            print(chess.square_name(from_square), chess.square_name(to_square))
            print("- - - - - - -")


    def get_piece_model(self, piece_moved):
        piece_moved = str(piece_moved).lower()
        if piece_moved == 'p':
            return self.pawn
        elif piece_moved == 'b':
            return self.bishop
        elif piece_moved == 'n':
            return self.knight
        elif piece_moved == 'r':
            return self.rook
        elif piece_moved == 'q':
            return self.queen
        elif piece_moved == 'k':
            return self.king
        else:
            print("invalid input")


    def get_from_square(self, fen, white_turn):

        board_state = torch.from_numpy(fen_to_board(fen=fen, white_turn=white_turn))[None, :]
        board = chess.Board(fen=fen)

        from_square_probs = F.softmax(self.selector(board_state), dim=1)
        from_square_probs = torch.sort(from_square_probs, descending=True)

        for square in from_square_probs.indices[0]:
            if board.piece_at(int(square)) is not None:
                from_square = square
                piece_moved = board.piece_at(int(square))
                break
        else:
            print("Cannot find legal from_square.")
            return False
        return from_square, piece_moved


    def get_to_square(self, fen, white_turn, from_square, piece_moved):

        model = self.get_piece_model(piece_moved=piece_moved)

        board_state = torch.from_numpy(fen_to_board(fen=fen, piece_values=PIECE_VALUES, white_turn=white_turn))[None, :]
        board = chess.Board(fen=fen)

        to_square_probs = F.softmax(model(board_state), dim=1)
        to_square_probs = torch.sort(to_square_probs, descending=True)

        for square in to_square_probs.indices[0]:
            if board.is_legal(chess.Move(int(from_square), int(square))):
                to_square = square
                break
        else:
            print("Cannot find legal to_square.")
            return False
        return to_square
            

if __name__=="__main__":

    selector_path = get_model_path(MODELS_DIR, 'selector')
    p_path = get_model_path(MODELS_DIR, 'p')
    b_path = get_model_path(MODELS_DIR, 'b')
    n_path = get_model_path(MODELS_DIR, 'n')
    r_path = get_model_path(MODELS_DIR, 'r')
    q_path = get_model_path(MODELS_DIR, 'q')
    k_path = get_model_path(MODELS_DIR, 'k')

    print("Initialising JoshFish 1.0")
    engine = Engine(selector_path, p_path, b_path, n_path, r_path, q_path, k_path)
    engine.play()

