from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.game.heavy_action import HeavyAction
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from typing import Callable
import math
import heapq
import random
import time


class MyPlayer(PlayerHex):

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self.nodes_explored = 0

    def halpha_beta_strategy(self, currentState: GameState, heuristic: Callable, time_limit: float = 1.0):
        self.nodes_explored = 0

        start_time = time.time()
        deadline = start_time + time_limit

        def timed_out():
            return time.time() >= deadline


        def max_value(state: GameState, alpha, beta, depth):
            if timed_out():
                raise TimeoutError
            self.nodes_explored += 1

            if state.is_done() or depth == 0:
                return heuristic(state), None

            v_star = -math.inf
            m_star = None

            for a in state.get_possible_heavy_actions():
                next_state = a.get_next_game_state()
                v, _ = min_value(next_state, alpha, beta, depth - 1)

                if v > v_star:
                    v_star = v
                    m_star = a
                    alpha = max(alpha, v_star)

                if v_star >= beta:
                    return v_star, m_star

            return v_star, m_star

        def min_value(state: GameState, alpha, beta, depth):
            if timed_out():
                raise TimeoutError
            self.nodes_explored += 1

            if state.is_done() or depth == 0:
                return heuristic(state), None

            v_star = math.inf
            m_star = None

            for a in state.get_possible_heavy_actions():
                next_state = a.get_next_game_state()
                v, _ = max_value(next_state, alpha, beta, depth - 1)

                if v < v_star:
                    v_star = v
                    m_star = a
                    beta = min(beta, v_star)

                if v_star <= alpha:
                    return v_star, m_star

            return v_star, m_star

        best_move_global = None
        depth = 1
        max_depth_reached = 0

        while True:
            try:
                value, best_move = max_value(currentState, -math.inf, math.inf, depth)

                if best_move is not None:
                    best_move_global = best_move

                max_depth_reached = depth
                depth += 1

            except TimeoutError:
                break
            except Exception:
                break

            if timed_out():
                break

        return best_move_global, max_depth_reached


    def heuristic(self, state):
        board = state.get_rep().env
        rows, cols = state.get_rep().get_dimensions()

        grid = [
            [board.get((i, j)).owner_id if board.get((i, j)) else None for j in range(cols)]
            for i in range(rows)
        ]

        player_id = self.id
        all_players = [p.id for p in state.players]
        adversary_id = [pid for pid in all_players if pid != player_id][0]

        directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

        def dijkstra_distance(start_cells, target_side, pid):
            dist = [[float('inf')] * cols for _ in range(rows)]
            pq = []

            for x, y in start_cells:
                if grid[x][y] == pid:
                    dist[x][y] = 0
                elif grid[x][y] is None:
                    dist[x][y] = 1
                else:
                    continue
                heapq.heappush(pq, (dist[x][y], x, y))

            while pq:
                d, x, y = heapq.heappop(pq)

                if target_side(x, y):
                    return d

                if d > dist[x][y]:
                    continue

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if grid[nx][ny] == pid:
                            cost = 0
                        elif grid[nx][ny] is None:
                            cost = 1
                        else:
                            cost = 1000

                        new_dist = d + cost
                        if new_dist < dist[nx][ny]:
                            dist[nx][ny] = new_dist
                            heapq.heappush(pq, (new_dist, nx, ny))

            return float('inf')

        player = next(p for p in state.players if p.id == player_id)
        adversary = next(p for p in state.players if p.id == adversary_id)

        if player.piece_type == "R":
            start_current = [(0, j) for j in range(cols)]
            target_current = lambda x, y: x == rows - 1
        else:
            start_current = [(i, 0) for i in range(rows)]
            target_current = lambda x, y: y == cols - 1

        if adversary.piece_type == "R":
            start_adv = [(0, j) for j in range(cols)]
            target_adv = lambda x, y: x == rows - 1
        else:
            start_adv = [(i, 0) for i in range(rows)]
            target_adv = lambda x, y: y == cols - 1

        dist_current = dijkstra_distance(start_current, target_current, player_id)
        dist_adv = dijkstra_distance(start_adv, target_adv, adversary_id)

        distance_score = dist_adv - dist_current


        center_score = 0
        for i in range(max(0, rows // 2 - 1), min(rows, rows // 2 + 2)):
            for j in range(max(0, cols // 2 - 1), min(cols, cols // 2 + 2)):
                if grid[i][j] == player_id:
                    center_score += 2
                elif grid[i][j] == adversary_id:
                    center_score -= 2

        block_score = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == adversary_id:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] is None:
                            block_score -= 1

        final_score = distance_score * 10 + center_score * 0.5 + block_score * 0.2

        return final_score

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        best_action, max_depth = self.halpha_beta_strategy(
            current_state,
            self.heuristic,
            time_limit=8.6
        )

        print(f"[{self.name}] Nœuds explorés : {self.nodes_explored} | Profondeur max : {max_depth}")

        if best_action is None:
            possible = list(current_state.get_possible_heavy_actions())
            return random.choice(possible)

        return best_action
