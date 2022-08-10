import numpy as np
import asyncio
import chess
import chess.engine
from engine import Engine
from IPython.display import display
from config import *
from functions import *
import json
import math
import random
from tqdm import tqdm
import sys
import pathlib
import argparse


NUM_GAMES = 100

parser = argparse.ArgumentParser()
parser.add_argument('conf', default='./sunfish/engines.json', nargs='?',
                    help='Location of engines.json file to use')
parser.add_argument('name', default='sunfish', nargs='?', help='Name of engine to use')
parser.add_argument('-selfplay', action='store_true', help='Play against itself')
parser.add_argument('-movetime', type=int, default=0, help='Movetime in ms')
parser.add_argument('-nodes', type=int, default=0, help='Maximum nodes')
parser.add_argument('-pvs', nargs='?', const=3, default=0, type=int,
                    help='Show Principal Variations (when mcts)')


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


async def load_engine(engine_args, name):
    args = next(a for a in engine_args if a['name'] == name)
    curdir = str(pathlib.Path(__file__).parent)
    popen_args = {}
    if 'workingDirectory' in args:
        popen_args['cwd'] = args['workingDirectory'].replace('$FILE', curdir)
    cmd = args['command'].split()
    if cmd[0] == '$PYTHON':
        cmd[0] = sys.executable
    if args['protocol'] == 'uci':
        _, engine = await chess.engine.popen_uci(cmd, **popen_args)
    elif args['protocol'] == 'xboard':
        _, engine = await chess.engine.popen_xboard(cmd, **popen_args)

    await engine.configure({opt['name']: opt['value'] for opt in args.get('options', [])})
    return engine


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


def get_mockfish_move(mockfish, board):
    t0=time.time()
    move = mockfish.predict_move(fen=board.fen(), white_turn=board.turn)
    t1 = time.time()
    time_taken = t1-t0
    return move, time_taken


async def get_engine_move(engine, board, limit, game_id):
    play_result = await engine.play(board, limit, game=game_id)
    return play_result.move


async def play(mockfish, engine):

    results = []

    for colour in tqdm([chess.WHITE, chess.BLACK]):

        for _ in tqdm(range(NUM_GAMES//2)):

            board = chess.Board()

            game_id = random.random()

            while not board.is_game_over():
                #print_unicode_board(board, perspective=colour)
                if colour == board.turn:
                    move, time_taken = get_mockfish_move(mockfish=mockfish, board=board)
                    time_limit = chess.engine.Limit(time=time_taken)
                else:
                    move = await get_engine_move(engine=engine, board=board, limit=time_limit, game_id=game_id)
                    #print(f' Sunfish move: {board.san(move)}')
                board.push(move)

            # Print status
            #print_unicode_board(board, perspective=colour)
            #print('Result:', board.result())
            results.append(board.result())
     
    print("white results:")
    print(results[:NUM_GAMES//2])
    print("black results:")
    print(results[NUM_GAMES//2:])
    return results


async def main():

    args = parser.parse_args()
    conf = json.load(open(args.conf))

    engine = await load_engine(conf, args.name)
    if 'author' in engine.id:
        print(f"Playing against {engine.id['name']} by {engine.id['author']}.")
    else:
        print(f"Playing against {engine.id['name']}.")
    
    mockfish = load_mockfish()

    try:
        await play(mockfish, engine)
    finally:
        print('\nGoodbye!')
        await engine.quit()


asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass