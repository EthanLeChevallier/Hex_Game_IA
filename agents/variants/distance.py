from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math
import random 
from typing import Callable
import heapq

class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)


    def halpha_beta_strategy(self, currentState: GameState, heuristic : Callable):

            def max_value(state: GameState, alpha, beta, depth):
                if state.is_done() or depth == 0:
                    return heuristic(state), None
                v_star = - math.inf
                m_star = None

                for a in state.get_possible_heavy_actions():
                    next_state = a.get_next_game_state()
                    v, _ = min_value(next_state, alpha, beta, depth-1)
                    if v > v_star:
                        v_star = v
                        m_star = a
                        alpha = max(alpha, v_star)
                    if v_star >= beta:
                        return v_star, m_star
                    
                return (v_star, m_star)
            
            def min_value(state: GameState, alpha, beta, depth):
                if state.is_done() or depth == 0:
                    return heuristic(state), None
                v_star = math.inf
                m_star = None
                
                for a in  state.get_possible_heavy_actions():
                    next_state = a.get_next_game_state()
                    v, _ = max_value(next_state, alpha, beta, depth-1)
                    if v < v_star:
                        v_star = v
                        m_star = a
                        beta = min(beta, v_star)
                    if v_star <= alpha:
                        return v_star, m_star
                    
                return (v_star, m_star)
            
            return max_value(currentState, -math.inf, math.inf, 2)[1]
    
    def heuristic(self, state):
        board = state.get_rep().env
        rows, cols = state.get_rep().get_dimensions()
        grid = [[board.get((i, j)).owner_id if board.get((i, j)) else None for j in range(cols)] for i in range(rows)]

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

        # Configuration des joueurs
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

        # Calcul des distances
        dist_current = dijkstra_distance(start_current, target_current, player_id)
        dist_adv = dijkstra_distance(start_adv, target_adv, adversary_id)

        # Score simple : différence de distances
                # Score principal : différence de distances
        distance_score = dist_adv - dist_current
        
        return distance_score


    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        boards = current_state.get_rep()
        board = current_state.get_rep().env


        # Premier coup : aucun pion sur le plateau 
        if  current_state.get_step() <= 1:
            rows, cols = current_state.get_rep().get_dimensions()
            
            center_cells = [
                (x, y)
                for x in range(rows // 2 - 2, rows // 2 + 1)
                for y in range(cols // 2 - 2, cols // 2 + 1)
            ]

            chosen_cell = random.choice(center_cells)
            print(chosen_cell)
            for a in current_state.get_possible_heavy_actions():
                next_env = a.get_next_game_state().get_rep().env
                if chosen_cell in next_env and next_env[chosen_cell].owner_id == self.get_id():
                    return a
            # --- Vérifier si un coup peut gagner immédiatement ---
        for a in current_state.get_possible_heavy_actions():
            next_state = a.get_next_game_state()
            if next_state.is_done() :
                return a 
            
        # Sinon : Minimax habituel
        return self.halpha_beta_strategy(current_state, heuristic=self.heuristic)