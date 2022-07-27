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


    def play(self, colour='w'):

        board = chess.Board()
        fen = board.fen()
        white_turn = board.turn

        if colour=='b':
            comp_move = self.predict_move(fen=fen, white_turn=board.turn)
            board.push(comp_move)

        while not board.is_game_over():

            display(chess.svg.board(board, size=500, flipped = not board.turn))
            #print(board)

            move_is_legal = False
            while not move_is_legal:
                user_move = input("Your move: ")
                try:
                    board.push_san(user_move)
                    move_is_legal = True
                except ValueError:
                    print(f"Illegal move: {user_move}")

            fen = board.fen()
            comp_move = self.predict_move(fen=fen, white_turn=board.turn)
            try:
                board.push(comp_move)
            except:
                display(chess.svg.board(board, size=500, flipped = board.turn))
                print(f"checkmate: {board.is_game_over()}")



    def predict_move(self, fen, white_turn):

        board = chess.Board(fen=fen)
        board.turn = white_turn
        board_state = torch.from_numpy(fen_to_board(fen=fen, white_turn=white_turn))[None, :]

        from_square_probs = F.softmax(self.selector(board_state), dim=1)
        from_square_probs = torch.sort(from_square_probs, descending=True)
        from_squares = from_square_probs.indices[0]

        if not white_turn:
            from_squares = 63 * torch.ones(len(from_squares)) - from_squares

        for square in from_squares:
            if board.piece_at(int(square)) is not None:

                from_square = square
                piece_moved = board.piece_at(int(square))

                to_square_model = self.get_piece_model(piece_moved=piece_moved)
                to_square_probs = F.softmax(to_square_model(board_state), dim=1)
                to_square_probs = torch.sort(to_square_probs, descending=True)
                to_squares = to_square_probs.indices[0]

                if not white_turn:
                    to_squares = 63 * torch.ones(len(to_squares)) - to_squares

                for square in to_squares:
                    if board.is_legal(chess.Move(int(from_square), int(square))):
                        to_square = square
                        if str(board.piece_at(int(from_square))).lower() == 'p':
                            promotion = 5
                        else:
                            promotion = None
                        return chess.Move(
                            int(from_square), int(to_square), 
                            promotion=promotion if int(to_square)>56 else None)
    

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



if __name__=="__main__":
    
    selector_path = get_model_path(MODELS_DIR, 'selector')
    p_path = get_model_path(MODELS_DIR, 'p')
    b_path = get_model_path(MODELS_DIR, 'b')
    n_path = get_model_path(MODELS_DIR, 'n')
    r_path = get_model_path(MODELS_DIR, 'r')
    q_path = get_model_path(MODELS_DIR, 'q')
    k_path = get_model_path(MODELS_DIR, 'k')
    
    engine = Engine(selector_path, p_path, b_path, n_path, r_path, q_path, k_path)
    engine.play(colour='w')

