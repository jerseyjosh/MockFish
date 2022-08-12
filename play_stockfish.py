import numpy as np
import chess
import chess.engine
from engine import Engine
from IPython.display import display
from config import *
from functions import *
from tqdm import tqdm


def print_unicode_board(board, perspective=chess.WHITE):
    """ Prints the position from a given perspective. """
    sc, ec = '\x1b[0;30;107m', '\x1b[0m'
    for r in range(8) if perspective == chess.BLACK else range(7, -1, -1):
        line = [f'{sc} {r+1}']
        for c in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            color = '\x1b[48;5;255m' if (r + c) % 2 == 1 else '\x1b[48;5;253m'
            if board.move_stack:
                if board.move_stack[-1].to_square == 8 * r + c:
                    color = '\x1b[48;5;153m'
                elif board.move_stack[-1].from_square == 8 * r + c:
                    color = '\x1b[48;5;153m'
            piece = board.piece_at(8 * r + c)
            line.append(color +
                        (chess.UNICODE_PIECE_SYMBOLS[piece.symbol()] if piece else ' '))
        print(' ' + ' '.join(line) + f' {sc} {ec}')
    if perspective == chess.WHITE:
        print(f' {sc}   a b c d e f g h  {ec}\n')
    else:
        print(f' {sc}   h g f e d c b a  {ec}\n')


def load_mockfish():
    selector_path = get_model_path(MODELS_DIR, 'selector')
    p_path = get_model_path(MODELS_DIR, 'p')
    b_path = get_model_path(MODELS_DIR, 'b')
    n_path = get_model_path(MODELS_DIR, 'n')
    r_path = get_model_path(MODELS_DIR, 'r')
    q_path = get_model_path(MODELS_DIR, 'q')
    k_path = get_model_path(MODELS_DIR, 'k')
    mockfish = Engine(selector_path, p_path, b_path, n_path, r_path, q_path, k_path)
    return mockfish


def get_mockfish_move(mockfish, board):
    t0=time.time()
    move = mockfish.predict_move_probabilistic(fen=board.fen(), white_turn=board.turn)
    t1 = time.time()
    time_taken = t1-t0
    return move, time_taken


def play(stockfish_path, level, num_games):

    winning_games = []

    mockfish = load_mockfish()

    #engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    #engine.configure({"Skill Level": STOCKFISH_LEVELS[level]["skill"]})
    engine = chess.engine.SimpleEngine.popen_xboard("./engines/stockfish1.exe")
    print("Stockfish settings:")
    print(STOCKFISH_LEVELS[level])

    limit = chess.engine.Limit(time=STOCKFISH_LEVELS[level]['time'], depth=STOCKFISH_LEVELS[level]["depth"])

    results = []

    for colour in [chess.WHITE, chess.BLACK]:

        for i in range(num_games//2):

            board = chess.Board()

            while not board.is_game_over():
                if colour == board.turn:
                    move, time_taken = get_mockfish_move(mockfish=mockfish, board=board)
                    limit = chess.engine.Limit(time=STOCKFISH_LEVELS[level]['time'], depth=STOCKFISH_LEVELS[level]["depth"])
                else:
                    move = engine.play(board, limit).move
                if board.is_legal(move):
                    board.push(move)
                else:
                    print('error')
                    break
            if board.outcome():
                if board.outcome().winner is not None:
                    if not (board.outcome().winner ^ colour):
                        winning_games.append([str(move) for move in board.move_stack])
                    print(not (board.outcome().winner ^ colour))
                    results.append(not (board.outcome().winner ^ colour))
                else:
                    print(None)
                    results.append(None)
            else:
                print(False)
                results.append(False)

    engine.quit()     

    results = [int(result) if result is not None else 0.5 for result in results]
    
     
    return results, winning_games


if __name__=="__main__":

    NUM_GAMES = 100

    levels = np.arange(1, 9)
    win_rates = []
    draw_rates = []
    lose_rates = []

    for level in levels:
        results, winning_games = play("./stockfish", level=level, num_games=NUM_GAMES)
        win_rates.append(results.count(1) / len(results))
        draw_rates.append(results.count(0.5) / len(results))
        lose_rates.append(results.count(0) / len(results))

    df = pd.DataFrame({"levels": levels, "win_rates": win_rates, "draw_rates": draw_rates, "lose_rates": lose_rates})
    df.to_csv(RESULTS_DIR + "stockfish_win_rates.csv")
    print(df)
