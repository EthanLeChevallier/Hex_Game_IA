from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.game.heavy_action import HeavyAction
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math
import random 
from typing import Callable
import heapq
import time



class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._dead_cells = set()
        self._dijkstra_cache = {}  
        self._transposition_table = {} 
        self._zobrist_table = None  
        self._zobrist_initialized = False
        self._dead_cell_cache = {}  

    def init_zobrist(self, rows, cols, num_players=2):
        """Crée une table Zobrist de dimensions [rows][cols][num_players + 1]."""
        random.seed(42) 
        self._zobrist_table = [[[random.getrandbits(64) for _ in range(num_players + 1)]
                                for _ in range(cols)] for _ in range(rows)]
        self._zobrist_initialized = True

    def zobrist_hash(self, grid):
        """Calcule un hash 64 bits unique pour une configuration du plateau."""
        if not self._zobrist_initialized:
            rows, cols = len(grid), len(grid[0])
            self.init_zobrist(rows, cols)

        h = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                cell = grid[i][j]
                if cell is None:
                    cell_index = 0 
                elif cell == self.id:
                    cell_index = 1  
                else:
                    cell_index = 2  
                h ^= self._zobrist_table[i][j][cell_index]
        return h
    
    def lookup_transposition(self, zobrist_key, depth, alpha, beta):
        entry = self._transposition_table.get(zobrist_key)
        if entry is not None and entry['depth'] >= depth:
            if entry['flag'] == 'EXACT':
                return entry['value'], entry.get('best_move'), alpha, beta
            elif entry['flag'] == 'LOWERBOUND' and entry['value'] > alpha:
                alpha = entry['value']
            elif entry['flag'] == 'UPPERBOUND' and entry['value'] < beta:
                beta = entry['value']
            return None, entry.get('best_move'), alpha, beta
        return None, None, alpha, beta

    def store_transposition(self, zobrist_key, depth, value, flag, best_move=None):
        self._transposition_table[zobrist_key] = {
            'value': value,
            'depth': depth,
            'flag': flag,
            'best_move': best_move
        }

    def rotations(self,lst):
        """Retourne les 6 rotations circulaires d'une liste de 6 éléments."""
        return [lst[i:] + lst[:i] for i in range(6)]


    def matches_pattern(self, neighbors, pattern):
        """Vérifie si la liste neighbors correspond à une rotation du pattern"""
        for rotated in self.rotations(pattern):
            match = True
            for n, p in zip(neighbors, rotated):
                if p != "." and n != p:
                    match = False
                    break
            if match:
                return True
        return False

    def is_dead_cell(self,grid, x, y, player_id, directions):
        """"E" = enemy, "A" = ally, "." = empty, "B" = border"""
        rows, cols = len(grid), len(grid[0])
        if grid[x][y] is not None:
            return False

        cache_key = (tuple(tuple(row) for row in grid), x, y, player_id)

        if cache_key in self._dead_cell_cache:
            self._dead_cache_hits += 1
            return self._dead_cell_cache[cache_key]
    
        # Construction du voisinage
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                neighbors.append("B")
            elif grid[nx][ny] is None:
                neighbors.append(".")
            elif grid[nx][ny] == player_id:
                neighbors.append("A")
            else:
                neighbors.append("E")

        # Tous les motifs de "cellules mortes"
        patterns = [
            # --- Motifs “intérieurs” ---
            ["E", "E", "E", "E", ".", "."],
            ["E", "E", "E", ".", "A", "."],
            ["E", "E", ".", "A", "A", "."],

            # --- Motifs “bords et coins” ---
            ["B", "E", "E", "B", ".", "."],    
            ["B", "E", ".", "E", "B", "."],    
            ["B", "E", ".", ".", "A", "B"],     
            ["B", "B", "E", "E", ".", "."],     
            ["B", "B", "B", "E", ".", "."],     
        ]

        result = False  
        for pattern in patterns:
            if self.matches_pattern(neighbors, pattern):
                result = True
                break
        
        self._dead_cell_cache[cache_key] = result
        return result

    def update_dead_cells(self, state: GameState):
        board = state.get_rep().env
        rows, cols = state.get_rep().get_dimensions()
        grid = [[board.get((i, j)).owner_id if board.get((i, j)) else None for j in range(cols)] for i in range(rows)]
        
        player_id = self.id
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
        
        self._dead_cells.clear()
        
        for i in range(rows):
            for j in range(cols):
                if self.is_dead_cell(grid, i, j, player_id, directions):
                    self._dead_cells.add((i, j))

    def get_action_position(self, action : HeavyAction) -> tuple[int, int] | None:
        """Retourne la position (i, j) jouée entre deux états, ou None si non trouvée."""
        before = action.get_current_game_state().get_rep().env
        after = action.get_next_game_state().get_rep().env

        rows, cols = action.get_current_game_state().get_rep().get_dimensions()
        for i in range(rows):
            for j in range(cols):
                cell_before = before.get((i, j))
                cell_after = after.get((i, j))
                if cell_before is None and cell_after is not None:
                    return (i, j)  # case où le joueur vient de poser une pièce
        return None

    def get_ordered_actions(self, state: GameState):
        actions_with_priority = []
        rows, cols = state.get_rep().get_dimensions()
        board = state.get_rep().env
        grid = [[board.get((i, j)).owner_id if board.get((i, j)) else None for j in range(cols)] for i in range(rows)]
        player_id = self.id
        adversary_id = [p.id for p in state.players if p.id != player_id][0]
        player = next(p for p in state.players if p.id == player_id)
        
        # Compter le nombre total de pièces posées pour détecter le premier coup
        total_pieces = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] is not None:
                    total_pieces += 1
        
        is_first_move = (total_pieces == 0)

        for a in state.get_possible_heavy_actions():
            pos = self.get_action_position(a)
            if not pos or pos in self._dead_cells:
                continue

            i, j = pos
            priority = 0

            # 1. Priorité au centre UNIQUEMENT au premier coup
            if is_first_move:
                center_i, center_j = rows // 2, cols // 2
                dist_to_center = abs(i - center_i) + abs(j - center_j)
                priority += (rows + cols - dist_to_center) * 10 

            # 2. Priorité aux cases proches de tes pièces 
            ally_adjacent = 0
            for dx, dy in [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < rows and 0 <= nj < cols:
                    if grid[ni][nj] == player_id:
                        ally_adjacent += 1
            priority += ally_adjacent * 5

            # 3. Priorité aux cases qui bloquent l'adversaire
            enemy_adjacent = 0
            for dx, dy in [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < rows and 0 <= nj < cols:
                    if grid[ni][nj] == adversary_id:
                        enemy_adjacent += 1
            priority += enemy_adjacent * 3

            actions_with_priority.append((priority, a))

        # Trier par priorité décroissante
        actions_with_priority.sort(key=lambda x: -x[0])
        
        # Retourner les meilleures actions (limité à 30)
        return [a for _, a in actions_with_priority[:22]]

    def halpha_beta_strategy(self, currentState: GameState, depth: int, heuristic: Callable):
        
        def max_value(state: GameState, alpha, beta, depth):
            if state.is_done() or depth == 0:
                return heuristic(state), None

            v_star = -math.inf
            m_star = None
            alpha_original = alpha

            # Zobrist hash pour cette position
            grid = [[state.get_rep().env.get((i,j)).owner_id if state.get_rep().env.get((i,j)) else None
                    for j in range(state.get_rep().get_dimensions()[1])]
                    for i in range(state.get_rep().get_dimensions()[0])]
            zobrist_key = self.zobrist_hash(grid)
            tt_value, tt_best_move, alpha, beta = self.lookup_transposition(zobrist_key, depth, alpha, beta)
            if tt_value is not None:
                return tt_value, tt_best_move
            
            ordered_actions = self.get_ordered_actions(state)

            for a in ordered_actions:
                pos = self.get_action_position(a)
                if pos in self._dead_cells:
                    continue

                next_state = a.get_next_game_state()
                v, _ = min_value(next_state, alpha, beta, depth - 1)

                if v > v_star:
                    v_star = v
                    m_star = a
                    alpha = max(alpha, v_star)
                if v_star >= beta:
                    break

            # Stocker dans la table de transposition
            flag = 'EXACT'
            if v_star <= alpha_original:
                flag = 'UPPERBOUND'
            elif v_star >= beta:
                flag = 'LOWERBOUND'
            self.store_transposition(zobrist_key, depth, v_star, flag, best_move=m_star)

            return v_star, m_star

        def min_value(state: GameState, alpha, beta, depth):
            if state.is_done() or depth == 0:
                return heuristic(state), None

            v_star = math.inf
            m_star = None
            beta_original = beta

            # Zobrist hash pour cette position
            grid = [[state.get_rep().env.get((i,j)).owner_id if state.get_rep().env.get((i,j)) else None
                    for j in range(state.get_rep().get_dimensions()[1])]
                    for i in range(state.get_rep().get_dimensions()[0])]
            zobrist_key = self.zobrist_hash(grid)
            tt_value, tt_best_move, alpha, beta = self.lookup_transposition(zobrist_key, depth, alpha, beta)
            if tt_value is not None:
                return tt_value, tt_best_move

            ordered_actions = self.get_ordered_actions(state)

            for a in ordered_actions:
                pos = self.get_action_position(a)
                if pos in self._dead_cells:
                    continue

                next_state = a.get_next_game_state()
                v, _ = max_value(next_state, alpha, beta, depth - 1)

                if v < v_star:
                    v_star = v
                    m_star = a
                    beta = min(beta, v_star)
                if v_star <= alpha:
                    break

            # Stocker dans la table de transposition
            flag = 'EXACT'
            if v_star <= alpha:
                flag = 'UPPERBOUND'
            elif v_star >= beta_original:
                flag = 'LOWERBOUND'
            self.store_transposition(zobrist_key, depth, v_star, flag, best_move=m_star)

            return v_star, m_star

        return max_value(currentState, -math.inf, math.inf, depth)[1]

    def heuristic(self, state):
        board = state.get_rep().env
        rows, cols = state.get_rep().get_dimensions()
        grid = [[board.get((i, j)).owner_id if board.get((i, j)) else None for j in range(cols)] for i in range(rows)]

        if state.is_done():
            # Déterminer le gagnant en regardant les scores
            player_id = self.id
            scores = state.scores
        
            if scores.get(player_id) == 1.0:
                return float('inf')
            elif any(score == 1.0 for pid, score in scores.items() if pid != player_id):
                return float('-inf')
            else:
                return 0 
    
        
        player_id = self.id
        all_players = [p.id for p in state.players]
        adversary_id = [pid for pid in all_players if pid != player_id][0]

        directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

        def dijkstra_distance(start_cells, target_side, pid, grid):
            zobrist_key = self.zobrist_hash(grid)
            cache_key = (zobrist_key, pid, target_side.__name__)

            if cache_key in self._dijkstra_cache:
                return self._dijkstra_cache[cache_key]
            
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
                    self._dijkstra_cache[cache_key] = d
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
            
            self._dijkstra_cache[cache_key] = float('inf')
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
        dist_current = dijkstra_distance(start_current, target_current, player_id,grid)
        dist_adv = dijkstra_distance(start_adv, target_adv, adversary_id,grid)

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
        self.update_dead_cells(current_state)
        
        if len(self._transposition_table) > 100000:
            self._transposition_table.clear()
        if len(self._dijkstra_cache) > 50000:
            self._dijkstra_cache.clear()
        if len(self._dead_cell_cache) > 50000:
            self._dead_cell_cache.clear()

        board = current_state.get_rep().env
        rows, cols = current_state.get_rep().get_dimensions()

        # compter cases vides
        playable_count = 0
        for i in range(rows):
            for j in range(cols):
                if board.get((i, j)) is None and (i, j) not in self._dead_cells:
                    playable_count += 1

        total = rows * cols
        playable_ratio = playable_count / total

        # choisir profondeur dynamique
        if playable_ratio > 0.50:
            depth = 4
        elif playable_ratio > 0.20:
            depth = 5
        else:
            depth = 6  # fin de partie

        start_time = time.time()
        action = self.halpha_beta_strategy(current_state, depth, heuristic=self.heuristic)
        end_time = time.time()
        print(f"[DEBUG] my player Temps de décision : {end_time - start_time:.6f} secondes, profondeur = {depth}")

        return action