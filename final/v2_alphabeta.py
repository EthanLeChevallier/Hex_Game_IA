from player_hex import PlayerHex
from seahorse.game.action import Action
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.game.light_action import LightAction
from seahorse.game.game_state import GameState
from typing import Optional
import time
import random


class MyPlayer(PlayerHex):
    """
    Minimal player implementing two search modes (plain minimax and alpha-beta),
    WITHOUT any heuristic (evaluation at non-terminal depth==0 returns 0).

    This player collects per-move and total statistics so you can compare
    the effect of alpha-beta pruning vs plain minimax.

    Usage:
    - Pass `use_alpha_beta=True` when constructing to enable alpha-beta.
    - Pass `max_depth` to set the search depth.

    Stats available after play:
    - nodes_per_move: list[int]
    - total_nodes: int
    - time_per_move: list[float] (seconds)
    - total_time: float (seconds)
    - moves_played: int
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer", *, use_alpha_beta: bool = False, max_depth: int = 2):
        super().__init__(piece_type, name)
        # Configuration
        self.use_alpha_beta = use_alpha_beta
        self.max_depth = max_depth

        # Stats (simple JSON-serializable containers)
        self.nodes_per_move = []     # nodes visited for each move
        self.total_nodes = 0         # total nodes visited across the game
        self.time_per_move = []      # time (s) for each move
        self.total_time = 0.0        # total time spent searching
        self.moves_played = 0        # number of moves where we searched

    # ---------- public helpers ----------
    def reset_stats(self):
        """Reset statistics (call before a new match if needed)."""
        self.nodes_per_move.clear()
        self.total_nodes = 0
        self.time_per_move.clear()
        self.total_time = 0.0
        self.moves_played = 0

    def get_stats(self) -> dict:
        """Return a dict summary of collected stats."""
        avg_nodes = (sum(self.nodes_per_move) / len(self.nodes_per_move)) if self.nodes_per_move else 0
        avg_time = (sum(self.time_per_move) / len(self.time_per_move)) if self.time_per_move else 0.0
        return {
            "nodes_per_move": list(self.nodes_per_move),
            "total_nodes": int(self.total_nodes),
            "avg_nodes_per_move": float(avg_nodes),
            "time_per_move": list(self.time_per_move),
            "total_time": float(self.total_time),
            "avg_time_per_move": float(avg_time),
            "moves_played": int(self.moves_played),
            "use_alpha_beta": bool(self.use_alpha_beta),
            "max_depth": int(self.max_depth),
        }

    # ---------- search helpers (no heuristic) ----------
    def _terminal_utility(self, state: GameState) -> int:
        """
        Terminal utility: +1 if this player wins, -1 if opponent wins, 0 otherwise.
        (No heuristic when depth==0: return 0)
        """
        if not state.is_done():
            return 0
        # scores is a dict keyed by player id
        my_score = state.scores.get(self.id, 0)
        # opponent id is the other player in players list
        opp_id = next((p.id for p in state.players if p.id != self.id), None)
        opp_score = state.scores.get(opp_id, 0) if opp_id is not None else 0
        if my_score == 1:
            return 1
        if opp_score == 1:
            return -1
        return 0

    def compute_action(self, current_state: GameState, **kwargs) -> LightAction:
        """
        Compute action using either plain minimax or alpha-beta (controlled by
        `self.use_alpha_beta`). Optional keyword args:
          - max_depth: override self.max_depth for this move
          - use_alpha_beta: override self.use_alpha_beta for this move

        This method updates stats: nodes_per_move, total_nodes, time_per_move, total_time.
        """
        max_depth = int(kwargs.get("max_depth", self.max_depth))
        use_ab = bool(kwargs.get("use_alpha_beta", self.use_alpha_beta))

        # per-move node counter
        nodes_this_move = 0

        # helper to increment node count from nested functions
        def _inc_node():
            nonlocal nodes_this_move
            nodes_this_move += 1

        # Plain minimax (returns value, action)
        def minimax(state: GameState, depth: int, maximizing: bool):
            _inc_node()
            # terminal or depth limit
            if state.is_done() or depth == 0:
                return self._terminal_utility(state), None

            best_action = None
            if maximizing:
                best_val = -float("inf")
                for a in state.get_possible_light_actions():
                    next_state = state.apply_action(a)
                    v, _ = minimax(next_state, depth - 1, False)
                    if v > best_val:
                        best_val = v
                        best_action = a
                        # short-circuit on win
                        if best_val == 1:
                            break
                return best_val, best_action
            else:
                best_val = float("inf")
                for a in state.get_possible_light_actions():
                    next_state = state.apply_action(a)
                    v, _ = minimax(next_state, depth - 1, True)
                    if v < best_val:
                        best_val = v
                        best_action = a
                        if best_val == -1:
                            break
                return best_val, best_action

        # Alpha-beta version
        def alphabeta(state: GameState, depth: int, alpha: float, beta: float, maximizing: bool):
            _inc_node()
            if state.is_done() or depth == 0:
                return self._terminal_utility(state), None

            best_action = None
            if maximizing:
                v = -float("inf")
                for a in state.get_possible_light_actions():
                    next_state = state.apply_action(a)
                    val, _ = alphabeta(next_state, depth - 1, alpha, beta, False)
                    if val > v:
                        v = val
                        best_action = a
                        if v == 1:
                            return v, best_action
                    alpha = max(alpha, v)
                    if alpha >= beta:
                        break
                return v, best_action
            else:
                v = float("inf")
                for a in state.get_possible_light_actions():
                    next_state = state.apply_action(a)
                    val, _ = alphabeta(next_state, depth - 1, alpha, beta, True)
                    if val < v:
                        v = val
                        best_action = a
                        if v == -1:
                            return v, best_action
                    beta = min(beta, v)
                    if alpha >= beta:
                        break
                return v, best_action

        # Run the search and time it
        start = time.time()
        # who is maximizing at root? If the next player in the state is us, we maximize
        maximizing_root = (current_state.next_player.id == self.id) if hasattr(current_state, "next_player") else True

        if use_ab:
            val, action = alphabeta(current_state, max_depth, -float("inf"), float("inf"), maximizing_root)
        else:
            val, action = minimax(current_state, max_depth, maximizing_root)

        elapsed = time.time() - start

        # update stats
        self.nodes_per_move.append(nodes_this_move)
        self.total_nodes += nodes_this_move
        self.time_per_move.append(elapsed)
        self.total_time += elapsed
        self.moves_played += 1

        # If no action found (shouldn't happen), pick a random legal move
        if action is None:
            possible = list(current_state.get_possible_light_actions())
            if not possible:
                # no legal moves
                return None
            action = random.choice(possible)

        return action