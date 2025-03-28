import multiprocessing
import pandas as pd
import chess.pgn
import io
from stockfish import Stockfish
import numpy as np


def eval_mate(mate):
    """Convert mate scores to a numerical value."""
    return np.sign(mate) * (-10000 * np.tanh(0.18 * abs(mate)) + 100000)


def analyse_fen(args):
    """Analyze a single FEN position using Stockfish."""
    i, fen, depth = args
    engine = Stockfish(
        path=r"/afs/inf.ed.ac.uk/user/s27/s2738314/chess/stockfish/stockfish-ubuntu-x86-64-avx2",
        depth=depth,
        parameters={"Threads": 2, "Hash": 1024},
    )
    engine.set_fen_position(fen)
    features = [
        x["value"] if x["type"] == "cp" else eval_mate(x["value"])
        for x in engine.get_evaluation_all_depths()
    ]
    return i, features[-1] if i == 0 else features


if __name__ == "__main__":
    depth = 20
    df = pd.read_csv("filtered_games.csv")

    for index, white_cheat, black_cheat, pgn_string, white_elo, black_elo in df.values.tolist():
        try:
            white_cheat = list(white_cheat[10:])
            black_cheat = list(black_cheat[10:])
            cheats = [item for pair in zip(white_cheat, black_cheat) for item in pair] + white_cheat[len(black_cheat):]
            pgn = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn)
            board = game.board()
            fens = []
            for move in game.mainline_moves():
                board.push(move)
                if board.is_checkmate():
                    cheats.pop()
                    break
                fens.append(board.fen())

            fens = fens[19:]
            evals_arr = np.zeros((depth + 1, len(fens) - 1))
            meta_data_arr = np.zeros((2, len(fens) - 1))
            elos = [white_elo if i % 2 == 0 else black_elo for i in range(len(fens) - 1)]
            meta_data_arr[0, :] = elos
            meta_data_arr[1, :] = cheats

            # Prepare arguments for multiprocessing
            args = [(i, fen, depth) for i, fen in enumerate(fens)]

            # Use multiprocessing to analyze FEN positions
            with multiprocessing.Pool() as pool:
                results = pool.map(analyse_fen, args)

            # Aggregate results into evals_arr
            for i, features in results:
                if i == 0:
                    evals_arr[-1, 0] = features
                else:
                    evals_arr[:-1, i - 1] = features
            result = np.vstack((evals_arr, meta_data_arr))
            np.savetxt(f'processed_new/{index}.csv', result.reshape(-1, result.shape[-1]), delimiter=',')
            print(index)
        except Exception as e:
            with open("errors.txt", "a") as myfile:
                myfile.write(f"\n{index}")
