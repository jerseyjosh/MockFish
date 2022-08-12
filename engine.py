from custom_torch_objects import Mockfish
import torch.nn.functional as F
import torch
import chess
import chess.svg
from IPython.display import SVG, display
from config import *
from functions import *

class Engine():
    def __init__(self, selector_path, p_path, b_path, n_path, r_path, q_path, k_path, params=BEST_PARAMS):

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


    def play(self, colour='w', probabilistic=True):
        
        if probabilistic:
            move_engine = self.predict_move_probabilistic
        else:
            move_engine = self.predict_move

        with torch.no_grad():

            board = chess.Board()
            fen = board.fen()
            white_turn = board.turn

            if colour=='b':
                comp_move = move_engine(fen=fen, white_turn=board.turn)
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
                comp_move = move_engine(fen=fen, white_turn=board.turn)
                try:
                    board.push(comp_move)
                except:
                    display(chess.svg.board(board, size=500, flipped = board.turn))
                    print(f"checkmate: {board.is_game_over()}")

    def predict_move_probabilistic(self, fen, white_turn):

        with torch.no_grad():

            board = chess.Board(fen=fen)
            board.turn = white_turn
            board_state = torch.from_numpy(fen_to_board(fen=fen, white_turn=white_turn))[None, :]

            from_square_scores = self.selector(board_state)
            pawn_scores = self.pawn(board_state)
            knight_scores = self.knight(board_state)
            bishop_scores = self.bishop(board_state)
            rook_scores = self.rook(board_state)
            queen_scores = self.queen(board_state)
            king_scores = self.king(board_state)


            best_moves = []
            best_scores = []

            for square, score in enumerate(from_square_scores.flatten()):

                if not white_turn:
                    square = 63-square

                if board.piece_at(square) and board.piece_at(square).symbol().isupper()==white_turn:

                    if board.piece_at(square).symbol().lower()=='p':
                        mask = torch.zeros(64)
                        mask[[move.to_square for move in board.legal_moves if move.from_square==square]] = 1
                        if not white_turn:
                            mask = mask.flip(dims=[0])
                        legal_pawn_scores = pawn_scores * mask
                        if legal_pawn_scores.any():
                            best_to_square = int(torch.argmax(legal_pawn_scores))
                            best_to_score = score * torch.max(legal_pawn_scores)
                            if not white_turn:
                                best_to_square = 63 - best_to_square
                            best_moves.append((square, best_to_square))
                            best_scores.append(best_to_score)
        
                    if board.piece_at(square).symbol().lower()=='b':
                        mask = torch.zeros(64)
                        mask[[move.to_square for move in board.legal_moves if move.from_square==square]] = 1
                        if not white_turn:
                            mask = mask.flip(dims=[0])
                        legal_bishop_scores = bishop_scores * mask
                        if legal_bishop_scores.any():
                            best_to_square = int(torch.argmax(legal_bishop_scores))
                            best_to_score = score * torch.max(legal_bishop_scores)
                            if not white_turn:
                                best_to_square = 63 - best_to_square
                            best_moves.append((square, best_to_square))
                            best_scores.append(best_to_score)

                    if board.piece_at(square).symbol().lower()=='n':
                        mask = torch.zeros(64)
                        mask[[move.to_square for move in board.legal_moves if move.from_square==square]] = 1
                        if not white_turn:
                            mask = mask.flip(dims=[0])
                        legal_knight_scores = knight_scores * mask
                        if legal_knight_scores.any():
                            best_to_square = int(torch.argmax(legal_knight_scores))
                            best_to_score = score * torch.max(legal_knight_scores)
                            if not white_turn:
                                best_to_square = 63 - best_to_square
                            best_moves.append((square, best_to_square))
                            best_scores.append(best_to_score)


                    if board.piece_at(square).symbol().lower()=='r':
                        mask = torch.zeros(64)
                        mask[[move.to_square for move in board.legal_moves if move.from_square==square]] = 1
                        if not white_turn:
                            mask = mask.flip(dims=[0])
                        legal_rook_scores = rook_scores * mask
                        if legal_rook_scores.any():
                            best_to_square = int(torch.argmax(legal_rook_scores))
                            best_to_score = score * torch.max(legal_rook_scores)
                            if not white_turn:
                                best_to_square = 63 - best_to_square
                            best_moves.append((square, best_to_square))
                            best_scores.append(best_to_score)

                    if board.piece_at(square).symbol().lower()=='q':
                        mask = torch.zeros(64)
                        mask[[move.to_square for move in board.legal_moves if move.from_square==square]] = 1
                        if not white_turn:
                            mask = mask.flip(dims=[0])
                        legal_queen_scores = queen_scores * mask
                        if legal_queen_scores.any():
                            best_to_square = int(torch.argmax(legal_queen_scores))
                            best_to_score = score * torch.max(legal_queen_scores)
                            if not white_turn:
                                best_to_square = 63 - best_to_square
                            best_moves.append((square, best_to_square))
                            best_scores.append(best_to_score)

                    if board.piece_at(square).symbol().lower()=='k':
                        mask = torch.zeros(64)
                        mask[[move.to_square for move in board.legal_moves if move.from_square==square]] = 1
                        if not white_turn:
                            mask = mask.flip(dims=[0])
                        legal_king_scores = king_scores * mask
                        if legal_king_scores.any():
                            best_to_square = int(torch.argmax(legal_king_scores))
                            best_to_score = score * torch.max(legal_king_scores)
                            if not white_turn:
                                best_to_square = 63 - best_to_square
                            best_moves.append((square, best_to_square))
                            best_scores.append(best_to_score)

            best_moves = np.array(best_moves)
            best_scores = F.softmax(torch.Tensor(best_scores), dim=0, dtype=torch.float64)

            chosen_move = best_moves[np.random.choice(len(best_moves), 1, p=best_scores)]
            from_square, to_square = chosen_move.flatten()

            if str(board.piece_at(int(from_square))).lower() == 'p':
                promotion = 5
            else:
                promotion = None
            move = chess.Move(from_square, to_square, promotion=promotion if to_square>55 or to_square<8 else None)
            return move


                    
    def predict_move_deterministic(self, fen, white_turn):

        with torch.no_grad():

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
                                promotion=promotion if int(to_square)>55 or to_square<8 else None)
                    

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
    board = chess.Board()

