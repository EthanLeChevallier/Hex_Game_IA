"""Run local headless matches and report optional player search statistics."""

import importlib
import os
import argparse
import csv
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

from board_hex import BoardHex
from game_state_hex import GameStateHex
from player_hex import PlayerHex
from seahorse.game.light_action import LightAction


def _import_module_from_path(path: str):
    folder = os.path.dirname(path)
    name = os.path.splitext(os.path.basename(path))[0]
    if folder and folder not in sys.path:
        sys.path.insert(0, folder)
    return importlib.import_module(name)


def run_match_local(p1: PlayerHex, p2: PlayerHex, dim: int = 7, max_steps: int = 200) -> dict:
    """Run a single match by stepping the game loop locally.

    Returns a summary dict containing stats for player 1 and 2 and the winner.
    """
    # reset stats on players if present
    if hasattr(p1, "reset_stats"):
        p1.reset_stats()
    if hasattr(p2, "reset_stats"):
        p2.reset_stats()

    init_scores = {p1.get_id(): 0, p2.get_id(): 0}
    env = {}
    init_rep = BoardHex(env=env, dim=[dim, dim])
    state = GameStateHex(scores=init_scores, next_player=p1, players=[p1, p2], rep=init_rep, step=0)

    steps = 0

    # We'll collect per-player runtime/node stats here in case the player
    # implementation doesn't expose get_stats(). We store per-move deltas.
    p1_nodes = []
    p1_times = []
    p1_moves = 0

    p2_nodes = []
    p2_times = []
    p2_moves = 0

    while not state.is_done() and steps < max_steps:
        current = state.next_player

        # snapshot node counters before calling compute_action
        pre_nodes = None
        if hasattr(current, "nodes_explored"):
            try:
                pre_nodes = int(getattr(current, "nodes_explored"))
            except Exception:
                pre_nodes = None
        elif hasattr(current, "total_nodes"):
            try:
                pre_nodes = int(getattr(current, "total_nodes"))
            except Exception:
                pre_nodes = None

        start_time = time.time()
        action = current.compute_action(state)
        elapsed = time.time() - start_time

        # snapshot after
        post_nodes = None
        if hasattr(current, "nodes_explored"):
            try:
                post_nodes = int(getattr(current, "nodes_explored"))
            except Exception:
                post_nodes = None
        elif hasattr(current, "total_nodes"):
            try:
                post_nodes = int(getattr(current, "total_nodes"))
            except Exception:
                post_nodes = None

        # compute delta nodes
        delta_nodes = 0
        if pre_nodes is not None and post_nodes is not None:
            delta_nodes = max(0, post_nodes - pre_nodes)
        else:
            # fallback: if player exposed nodes in print only, we cannot measure here
            delta_nodes = 0

        # record to the appropriate player bucket
        if current is p1:
            p1_nodes.append(delta_nodes)
            p1_times.append(elapsed)
            p1_moves += 1
        else:
            p2_nodes.append(delta_nodes)
            p2_times.append(elapsed)
            p2_moves += 1

        # Normalize action to LightAction if possible
        if action is None:
            # No legal actions (shouldn't happen) — break to avoid infinite loop
            break

        if not isinstance(action, LightAction):
            try:
                action = state.convert_heavy_action_to_light_action(action)
            except Exception:
                try:
                    action = LightAction(action)
                except Exception:
                    raise ValueError(f"Unsupported action type returned by player: {type(action)}")

        state = state.apply_action(action)
        steps += 1

    # determine winner: check scores
    winner = None
    try:
        if state.scores.get(p1.get_id(), 0) == 1:
            winner = p1.name
        elif state.scores.get(p2.get_id(), 0) == 1:
            winner = p2.name
    except Exception:
        winner = None

    # gather stats
    # If player exposes get_stats and it looks usable (non-empty totals), prefer it.
    def _usable_stats(d):
        try:
            return bool(d) and d.get("total_nodes", 0) != 0
        except Exception:
            return False

    s1 = p1.get_stats() if hasattr(p1, "get_stats") else {}
    if not _usable_stats(s1):
        total_nodes_1 = sum(p1_nodes)
        total_time_1 = sum(p1_times)
        avg_nodes_1 = (total_nodes_1 / len(p1_nodes)) if p1_nodes else 0.0
        avg_time_1 = (total_time_1 / len(p1_times)) if p1_times else 0.0
        s1 = {
            "nodes_per_move": list(p1_nodes),
            "total_nodes": int(total_nodes_1),
            "avg_nodes_per_move": float(avg_nodes_1),
            "time_per_move": list(p1_times),
            "total_time": float(total_time_1),
            "avg_time_per_move": float(avg_time_1),
            "moves_played": int(p1_moves),
        }

    s2 = p2.get_stats() if hasattr(p2, "get_stats") else {}
    if not _usable_stats(s2):
        total_nodes_2 = sum(p2_nodes)
        total_time_2 = sum(p2_times)
        avg_nodes_2 = (total_nodes_2 / len(p2_nodes)) if p2_nodes else 0.0
        avg_time_2 = (total_time_2 / len(p2_times)) if p2_times else 0.0
        s2 = {
            "nodes_per_move": list(p2_nodes),
            "total_nodes": int(total_nodes_2),
            "avg_nodes_per_move": float(avg_nodes_2),
            "time_per_move": list(p2_times),
            "total_time": float(total_time_2),
            "avg_time_per_move": float(avg_time_2),
            "moves_played": int(p2_moves),
        }

    return {"winner": winner, "steps": steps, "p1_stats": s1, "p2_stats": s2}


def run_experiments(my_player_path: str, opponents: list[str], matches_per_mode: int = 20, depth: int = 2, dim: int = 7, out_csv: str = "results.csv"):
    results = []

    for opp_path in opponents:
        mod_opp = _import_module_from_path(opp_path)
        opp_name = os.path.splitext(os.path.basename(opp_path))[0]

        for use_ab in (False, True):
            mode_name = "alpha_beta" if use_ab else "minimax"
            print(f"Running {matches_per_mode} matches vs {opp_name} with mode={mode_name}, depth={depth}")
            for m in range(matches_per_mode):
                # import fresh modules to avoid accidental state carryover
                my_mod = _import_module_from_path(my_player_path)
                opp_mod = _import_module_from_path(opp_path)

                # instantiate with minimal args to remain compatible with simplified MyPlayer
                p1 = my_mod.MyPlayer("R", name=f"me_{mode_name}")
                # set attributes if the class does not accept kwargs
                try:
                    setattr(p1, "use_alpha_beta", use_ab)
                except Exception:
                    pass
                try:
                    setattr(p1, "max_depth", depth)
                except Exception:
                    pass
                p2 = opp_mod.MyPlayer("B", name=f"{opp_name}")

                summary = run_match_local(p1, p2, dim=dim)

                row = {
                    "opponent": opp_name,
                    "mode": mode_name,
                    "match_idx": m,
                    "winner": summary.get("winner"),
                    "steps": summary.get("steps"),
                    # p1 stats
                    "p1_total_nodes": summary["p1_stats"].get("total_nodes", 0),
                    "p1_total_time": summary["p1_stats"].get("total_time", 0.0),
                    "p1_avg_nodes": summary["p1_stats"].get("avg_nodes_per_move", 0.0),
                    "p1_avg_time": summary["p1_stats"].get("avg_time_per_move", 0.0),
                    "p1_moves": summary["p1_stats"].get("moves_played", 0),
                }
                results.append(row)

    # write CSV
    keys = ["opponent", "mode", "match_idx", "winner", "steps", "p1_total_nodes", "p1_total_time", "p1_avg_nodes", "p1_avg_time", "p1_moves"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # print aggregated summary
    grouped = {}
    for r in results:
        k = (r["opponent"], r["mode"])
        grouped.setdefault(k, []).append(r)

    print("\nAggregated results:")
    for (opp, mode), rows in grouped.items():
        nodes = [row["p1_total_nodes"] for row in rows]
        times = [row["p1_total_time"] for row in rows]
        print(f"{opp} {mode}: matches={len(rows)}, avg_nodes={statistics.mean(nodes):.1f}, avg_time={statistics.mean(times):.4f}s")


def _default_opponents(here: str):
    return [os.path.join(here, "random_player_hex.py"), os.path.join(here, "greedy_player_hex.py")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run MyPlayer vs opponents and collect stats")
    parser.add_argument("--matches", type=int, default=20, help="matches per mode/opponent")
    parser.add_argument("--depth", type=int, default=2, help="search depth")
    parser.add_argument("--dim", type=int, default=7, help="board dimension (n for n x n)")
    parser.add_argument("--out", type=str, default="results.csv", help="output CSV file")
    parser.add_argument("--opponents", nargs="*", help="opponent module paths (default: random and greedy)")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("my_player", nargs="?", default=os.path.join(repo_root, "my_player.py"), help="path to your my_player.py")
    args = parser.parse_args()

    opponents = args.opponents if args.opponents else _default_opponents(repo_root)

    run_experiments(args.my_player, opponents, matches_per_mode=args.matches, depth=args.depth, dim=args.dim, out_csv=args.out)
