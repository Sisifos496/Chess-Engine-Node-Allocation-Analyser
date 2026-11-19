import argparse
import csv
import os
from pathlib import Path
import shutil
import time

import chess
import chess.engine

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "lichess_database.csv"
DEFAULT_OUTPUT = HERE / "analysis_results.csv"
DEFAULT_BUDGETS = (10_000, 20_000, 40_000, 80_000, 160_000, 320_000)

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    # Resolve project data and executable paths so the script works when
    # launched from outside this folder.
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="CSV of sampled positions with position_id, fen and phase columns")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish"),
                        help="Path to the Stockfish executable (or set STOCKFISH_PATH)")
    parser.add_argument("--positions", type=int, default=50,
                        help="Number of source positions to analyze (default: 50)")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)),
                        help="Comma-separated node budgets")
    args = parser.parse_args()
    if not args.stockfish:
        parser.error("Stockfish executable not found. Install Stockfish or provide --stockfish /path/to/stockfish")
    if not args.input.exists():
        parser.error(f"Position CSV not found: {args.input}")
    if args.positions < 1:
        parser.error("--positions must be positive")
    try:
        args.budgets = [int(part.strip()) for part in args.budgets.split(",")]
        if not args.budgets or any(b < 1 for b in args.budgets):
            raise ValueError
    except ValueError:
        parser.error("--budgets must be comma-separated positive integers")
    return args

def main():
    args = parse_arguments()
    # Read once and limit the work to the requested number of source positions.
    with args.input.open(newline="", encoding="utf-8-sig") as infile:
        positions = list(csv.DictReader(infile))[:args.positions]
    if not positions:
        raise SystemExit("The CSV is empty.")
    if not all("fen" in row and "position_id" in row for row in positions):
        raise SystemExit("Input CSV requires 'position_id' and 'fen' columns.")

    # position_id identifies the source position; phase and game_id preserve its
    # opening, middlegame, or endgame context and original game association.
    # fen contains the exact board state sent to Stockfish.
    # budget_nodes is the requested node limit; nodes_searched is the limit
    # actually reached, which can differ slightly due to search boundaries.
    # depth and seldepth are the main and selective search depths reached.
    # best_move_uci is the top move in universal chess notation; best_move_san
    # is the same move in human-readable notation for that board position.
    # eval_cp_stm is the centipawn evaluation from the side-to-move perspective;
    # mate_in_stm is the forced-mate distance from the side-to-move perspective.
    # engine_seconds and wall_seconds report reported search time and measured
    # elapsed time, respectively.
    # nps is the search speed in nodes per second; pv_uci lists the first moves
    # of the principal variation in UCI notation.
    # engine_name identifies the chess engine and version used for the analysis.
    columns = [
        "position_id", "phase", "game_id", "fen", "budget_nodes",
        "nodes_searched", "depth", "seldepth", "best_move_uci", "best_move_san",
        "eval_cp_stm", "mate_in_stm", "engine_seconds", "wall_seconds",
        "nps", "pv_uci", "engine_name"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        # A fixed one-thread configuration avoids introducing parallel search variability
        options = {key: value for key, value in {"Threads": 1, "Hash": 64}.items()
                   if key in engine.options}
        if options:
            engine.configure(options)
        engine_name = engine.id.get("name", "Stockfish")
        print(f"Engine: {engine_name}")
        print(f"Positions: {len(positions)} | Node budgets: {args.budgets}")

        with args.output.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=columns)
            writer.writeheader()
            for i, record in enumerate(positions, start=1):
                board = chess.Board(record["fen"])
                # Do not waste engine time on malformed positions or positions
                # that cannot provide a meaningful move search.
                if not board.is_valid() or board.is_game_over(claim_draw=True):
                    print(f"Skipping invalid or finished position {record['position_id']}")
                    continue

                for budget in args.budgets:
                    began = time.perf_counter()
                    # A unique game identifier resets engine game state and hash
                    # between analyses, so larger budgets don't inherit the
                    # smaller budget's transposition-table contents.
                    info = engine.analyse(
                        board,
                        chess.engine.Limit(nodes=budget),
                        game=object(),
                        info=chess.engine.INFO_ALL,
                    )
                    elapsed = time.perf_counter() - began
                    pv = info.get("pv", [])
                    best = pv[0] if pv else None
                    # Convert engine scores to side-to-move perspective so positive
                    # values consistently mean an advantage for the current player.
                    score = info.get("score")
                    relative = score.pov(board.turn) if score is not None else None
                    writer.writerow({
                        "position_id": record["position_id"],
                        "phase": record.get("phase", ""),
                        "game_id": record.get("game_id", ""),
                        "fen": record["fen"],
                        "budget_nodes": budget,
                        "nodes_searched": info.get("nodes", ""),
                        "depth": info.get("depth", ""),
                        "seldepth": info.get("seldepth", ""),
                        "best_move_uci": best.uci() if best else "",
                        "best_move_san": board.san(best) if best else "",
                        "eval_cp_stm": relative.score() if relative is not None else "",
                        "mate_in_stm": relative.mate() if relative is not None else "",
                        "engine_seconds": info.get("time", ""),
                        "wall_seconds": round(elapsed, 6),
                        "nps": info.get("nps", ""),
                        "pv_uci": " ".join(move.uci() for move in pv[:12]),
                        "engine_name": engine_name,
                    })
                    outfile.flush()  # Preserve completed rows if interrupted.

                print(f"[{i}/{len(positions)}] Analyzed {record['position_id']}", flush=True)
    finally:
        engine.quit()
    print(f"Finished. Results saved to: {args.output}")
    print("Note: These are shallow-search measurements, NOT verified move-quality labels.")

if __name__ == "__main__":
    main()
