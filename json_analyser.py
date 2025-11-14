#!/usr/bin/env python3
import json, sys

def get_hex_neighbors(r, c, board_size):
    """Retourne les voisins hexagonaux valides (coordonnées axiales)"""
    neighbors = []
    directions = [
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1)
    ]
    
    for dq, dr in directions:
        new_r, new_c = r + dr, c + dq
        if 0 <= new_r < board_size and 0 <= new_c < board_size:
            neighbors.append((new_r, new_c))
    return neighbors

def find_connected_component(start_r, start_c, board, piece_type, visited, board_size):
    """Trouve la composante connectée à partir d'une position donnée"""
    if (start_r, start_c) in visited or board[start_r][start_c] != piece_type:
        return set()
    
    component = set()
    stack = [(start_r, start_c)]
    
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
            
        visited.add((r, c))
        component.add((r, c))
        
        # Explorer les voisins hexagonaux
        for nr, nc in get_hex_neighbors(r, c, board_size):
            if (nr, nc) not in visited and board[nr][nc] == piece_type:
                stack.append((nr, nc))
    
    return component

def find_winning_chain(board, winner_piece):
    """Trouve et retourne la plus grande chaîne connectée du gagnant"""
    board_size = len(board)
    visited = set()
    largest_chain = set()
    
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] == winner_piece and (r, c) not in visited:
                component = find_connected_component(r, c, board, winner_piece, visited, board_size)
                if len(component) > len(largest_chain):
                    largest_chain = component
    
    return largest_chain

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 json_analyser.py <file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    # Récupérer les informations du gagnant depuis final_summary
    summary = data["final_summary"]
    players = summary["players"]
    winner_id = summary["winners_id"][0]
    
    # Créer un mapping des joueurs
    pmap = {p["id"]: p for p in players}
    winner_name = pmap[winner_id]["name"]
    
    # Déterminer la pièce du gagnant
    winner_piece = None
    for player in players:
        if player["id"] == winner_id:
            winner_piece = player.get("piece_type")
            break
    
    if winner_piece is None:
        #Chercher dans le dernier step piece_type
        last_step = data["steps"][-1]
        for player in last_step["players"]:
            if player["id"] == winner_id:
                winner_piece = player.get("piece_type")
                break
    
    print(f"Gagnant: {winner_name} (pièce: {winner_piece})")

    # Trouver le dernier step (le plus récent)
    last_step = data["steps"][-1]
    step_number = last_step["step"]
    
    print(f"Dernier step: {step_number}")

    # Récupérer les dimensions du plateau
    dims = last_step["rep"]["dim"]
    n = dims[0]

    # Initialiser le plateau vide
    board = [["." for _ in range(n)] for _ in range(n)]

    # Remplir le plateau avec les pièces du dernier état
    env = last_step["rep"]["env"]
    for coord_str, cell_info in env.items():
        # Convertir la clé de string "(r,c)" en tuple (r,c)
        if coord_str.startswith("(") and coord_str.endswith(")"):
            coord_str = coord_str[1:-1]  # Enlever les parenthèses
            r, c = map(int, coord_str.split(", "))
            board[r][c] = cell_info["piece_type"]

    # Trouver et afficher la chaîne gagnante
    if winner_piece:
        winning_chain = find_winning_chain(board, winner_piece)
        chain_size = len(winning_chain)
        
        print(f"\nChaîne gagnante ({winner_piece}):")
        print(f"Taille de la chaîne: {chain_size} pièces")
        

if __name__ == "__main__":
    main()