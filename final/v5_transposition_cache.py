from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.game.heavy_action import HeavyAction
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from typing import Callable
from collections import OrderedDict
import math
import heapq
import random 

class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._transposition = OrderedDict()
        self.max_table_size = 250000

        self.zobrist_table = None
        self.zobrist_initialized = False

        self._dijkstra_cache = {}
        self.dijkstra_hits = 0
        self.dijkstra_misses = 0

        self.nodes_explored = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.tt_stores = 0
        self.tt_replaces = 0
        self.tt_evictions = 0


    def init_zobrist(self, rows, cols, num_players=2):
        random.seed(42) 
        # Créer une entrée pour chaque joueur + case vide
        self.zobrist_table = [[[random.getrandbits(64) for _ in range(num_players + 1)]
                            for _ in range(cols)] for _ in range(rows)]
        self.zobrist_initialized = True

    def zobrist_hash(self, state: GameState):
        rep = state.get_rep()
        rows, cols = rep.get_dimensions()
        board = rep.env
        
        if not self.zobrist_initialized:
            self.init_zobrist(rows, cols, len(state.players))
        
        h = 0
        for i in range(rows):
            for j in range(cols):
                cell = board.get((i, j))
                if cell is None:
                    cell_index = 0  # Case vide
                else:
                    # Trouver l'index du joueur
                    cell_index = 0
                    for idx, player in enumerate(state.players, 1):
                        if player.id == cell.owner_id:
                            cell_index = idx
                            break
                    # Fallback si pas trouvé
                    if cell_index == 0:
                        cell_index = len(state.players)
                
                h ^= self.zobrist_table[i][j][cell_index]
        
        return h
    
    def halpha_beta_strategy(self, currentState: GameState, heuristic: Callable):
        # Reset stats
        self.nodes_explored = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.tt_stores = 0
        self.tt_replaces = 0
        self.tt_evictions = 0
        
        EXACT, LOWER, UPPER = 0, 1, 2

        def lookup(state_hash, depth, alpha, beta):
            if state_hash not in self._transposition:
                return None

            value, stored_depth, bound_type = self._transposition[state_hash]
            self.tt_hits += 1

            if stored_depth < depth:
                return None

            if bound_type == EXACT:
                return value

            if bound_type == LOWER:
                if value > alpha:
                    alpha = value
            elif bound_type == UPPER:
                if value < beta:
                    beta = value

            if alpha >= beta:
                self.tt_cutoffs += 1
                return value

            return None

        def store(state_hash, depth, alpha, beta, v_star):
            if v_star <= alpha:
                bound_type = UPPER
            elif v_star >= beta:
                bound_type = LOWER
            else:
                bound_type = EXACT

            entry = (v_star, depth, bound_type)

            if state_hash in self._transposition:
                self.tt_replaces += 1
                self._transposition[state_hash] = entry
                self._transposition.move_to_end(state_hash)
            else:
                self.tt_stores += 1
                self._transposition[state_hash] = entry

                if len(self._transposition) > self.max_table_size:
                    self.tt_evictions += 1
                    self._transposition.popitem(last=False)

        def max_value(state, alpha, beta, depth):
            self.nodes_explored += 1
            state_hash = self.zobrist_hash(state)
            
            # Check transposition table
            tt_value = lookup(state_hash, depth, alpha, beta)
            if tt_value is not None:
                self.nodes_explored += 1
                return tt_value, None

            # Terminal node or depth limit
            if state.is_done() or depth == 0:
                val = heuristic(state)
                return val, None

            v_star = -math.inf
            m_star = None
            possible_actions = list(state.get_possible_heavy_actions())
            
            for a in possible_actions:
                next_state = a.get_next_game_state()
                v, _ = min_value(next_state, alpha, beta, depth - 1)

                if v > v_star:
                    v_star = v
                    m_star = a

                alpha = max(alpha, v_star)
                if alpha >= beta:
                    break  # Beta cutoff

            store(state_hash, depth, alpha, beta, v_star)
            return v_star, m_star

        def min_value(state, alpha, beta, depth):
            self.nodes_explored += 1
            state_hash = self.zobrist_hash(state)
            
            # Check transposition table
            tt_value = lookup(state_hash, depth, alpha, beta)
            if tt_value is not None:
                self.nodes_explored += 1
                return tt_value, None

            # Terminal node or depth limit
            if state.is_done() or depth == 0:
                val = heuristic(state)
                return val, None

            v_star = math.inf
            m_star = None
            possible_actions = list(state.get_possible_heavy_actions())
            
            for a in possible_actions:
                next_state = a.get_next_game_state()
                v, _ = max_value(next_state, alpha, beta, depth - 1)

                if v < v_star:
                    v_star = v
                    m_star = a

                beta = min(beta, v_star)
                if alpha >= beta:
                    break  # Alpha cutoff

            store(state_hash, depth, alpha, beta, v_star)
            return v_star, m_star

        # Start the search
        best_value, best_action = max_value(currentState, -math.inf, math.inf, 2)
        return best_action

    def heuristic(self, state):
        board = state.get_rep().env
        rows, cols = state.get_rep().get_dimensions()
        grid = [[board.get((i, j)).owner_id if board.get((i, j)) else None for j in range(cols)] for i in range(rows)]

        player_id = self.id
        all_players = [p.id for p in state.players]
        adversary_id = [pid for pid in all_players if pid != player_id][0]

        directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

        def dijkstra_distance(start_cells, target_side, pid):
            key = (self.zobrist_hash(state), pid, target_side.__name__)
            if key in self._dijkstra_cache :
                self.dijkstra_hits += 1
                return self._dijkstra_cache [key]
            
            self.dijkstra_misses += 1
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
                    self._dijkstra_cache [key] = d
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

            self._dijkstra_cache [key] = float('inf')
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
        
        # Score additionnel : contrôle du centre
        center_score = 0
        center_cells = []
        for i in range(max(0, rows//2 - 1), min(rows, rows//2 + 2)):
            for j in range(max(0, cols//2 - 1), min(cols, cols//2 + 2)):
                if grid[i][j] == player_id:
                    center_score += 2
                elif grid[i][j] == adversary_id:
                    center_score -= 2
                elif grid[i][j] is None:
                    center_cells.append((i, j))

        # Score pour les coups qui bloquent l'adversaire
        block_score = 0
        # Vérifier les cases adjacentes aux pièces adverses importantes
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == adversary_id:
                    # Cette case appartient à l'adversaire
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] is None:
                            # Case vide adjacente à l'adversaire - potentiellement importante à bloquer
                            block_score -= 1

        # Heuristique finale avec pondérations
        final_score = distance_score * 10 + center_score * 0.5 + block_score * 0.2
    
        
        return final_score

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        best_action = self.halpha_beta_strategy(current_state, self.heuristic)

        if best_action is None:
            possible = list(current_state.get_possible_heavy_actions())
            print(f"[{self.name}] Nœuds explorés ce coup : {self.nodes_explored}")
            return random.choice(possible)

        print(f"[{self.name}] Nœuds explorés : {self.nodes_explored}   (positions évaluées dans l'arbre)")
        print(f"TT Hits      : {self.tt_hits}          (états retrouvés dans la table → calcul évité)")
        print(f"TT Cutoffs   : {self.tt_cutoffs}       (coupures provoquées grâce à un hit TT)")
        print(f"TT Stores    : {self.tt_stores}        (nouveaux états ajoutés dans la table)")
        print(f"TT Replaces  : {self.tt_replaces}      (anciens états écrasés par de plus profonds)")
        print(f"TT Evictions : {self.tt_evictions}     (entrées supprimées car la TT était pleine)")
        print(f"Taille TT    : {len(self._transposition)}  (taille actuelle de la table de transposition)")
        print("\n")
        print(f"Taille cache dijkstra      : {len(self._dijkstra_cache)}  (taille actuelle du cache)")
        print(f"Dijkstra hits : {self.dijkstra_hits}   (distances retrouvées dans le cache)")
        print(f"Dijkstra misses : {self.dijkstra_misses}  (distances calculées pour la première fois)")




        print(f"[{self.name}] Nœuds explorés ce coup : {self.nodes_explored}")
        return best_action