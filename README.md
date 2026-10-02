# Hex Game IA

Projet Python autour du jeu Hex : moteur de partie, interface graphique, agents de référence et plusieurs prototypes de recherche adversariale.

## Règles

La partie se joue sur un plateau hexagonal de 14 x 14 cases. Rouge joue en premier et cherche à relier le bord supérieur au bord inférieur. Bleu cherche à relier le bord gauche au bord droit. À tour de rôle, chaque joueur pose un pion sur une case libre. Le premier chemin continu entre ses deux bords fait gagner la partie.

## Prérequis et installation

Python 3.11 est la version utilisée dans les instructions d'origine. Depuis la racine du dépôt, créez un environnement virtuel et installez les dépendances :

```bash
python3.11 -m venv .venv
```

Activez-le, puis installez les dépendances :

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## Lancer une partie

Les commandes se lancent depuis la racine du dépôt.

Partie humain contre humain :

```bash
python main_hex.py -t human_vs_human
```

Deux agents locaux (aléatoire contre greedy) :

```bash
python main_hex.py -t local random_player_hex.py greedy_player_hex.py
```

Humain contre l'agent principal :

```bash
python main_hex.py -t human_vs_computer my_player.py
```

Pour jouer sans interface graphique, ajoutez `-g` :

```bash
python main_hex.py -t local -g random_player_hex.py greedy_player_hex.py
```

Les agents alternatifs du dossier `agents/variants/` peuvent être passés de la même manière, par exemple :

```bash
python main_hex.py -t local agents/variants/distance.py random_player_hex.py
```

Pour consulter toutes les options :

```bash
python main_hex.py --help
```

### Partie en réseau

Sur la machine hôte, remplacez `ADRESSE_IP` par l'adresse accessible depuis l'autre machine :

```bash
python main_hex.py -t host_game -a ADRESSE_IP my_player.py
```

Sur la machine qui rejoint la partie :

```bash
python main_hex.py -t connect -a ADRESSE_IP my_player.py
```

Le port par défaut est `16001` et peut être remplacé avec `-p`. Les connexions entre machines nécessitent que ce port soit accessible sur le réseau. Les modes réseau ne sont pas nécessaires pour les parties locales.

## Agents et expérimentations

Chaque fichier joueur expose une classe `MyPlayer` compatible avec le moteur. Les fichiers expérimentaux sont des variantes indépendantes : ils ne sont pas tous équivalents en force ou en maturité.

| Chemin | Rôle |
| --- | --- |
| `random_player_hex.py` | Joue un coup légal au hasard. |
| `greedy_player_hex.py` | Choisit un coup à partir d'un plus court chemin estimé. |
| `my_player.py` | Agent principal, avec recherche alpha-bêta et heuristiques. |
| `agents/variants/` | Prototypes alternatifs (distance, centre, cellules mortes et cache). |
| `final/` | Versions successives de recherche, de minimax/alpha-bêta à l'approfondissement itératif avec cache et ordonnancement. |

Les noms `v2_...`, `v3_...`, `v5_...`, `v6_...`, `v8_...` et `v9_...` dans `final/` identifient des étapes d'expérimentation; `play_and_report.py` lance des comparaisons et écrit `results.csv` dans le répertoire courant.

Exemple de benchmark réduit :

```bash
python final/play_and_report.py --matches 1 --depth 1 --dim 5
```

Le benchmark utilise un plateau de taille configurable, contrairement au lancement standard qui initialise le plateau officiel en 14 x 14. Pour sélectionner un autre joueur, passez son chemin en dernier argument; les adversaires peuvent être remplacés avec `--opponents`.

## Tests

Les tests utilisent la bibliothèque standard Python :

```bash
python -m unittest discover -v
```

## Structure

```text
.
├── agents/variants/   # Prototypes alternatifs
├── final/             # Variantes et outil de benchmark
├── GUI/               # Interface web et assets du plateau
├── tests/             # Tests du moteur
├── board_hex.py       # Plateau et voisinage hexagonal
├── game_state_hex.py  # État, actions et détection de victoire
├── main_hex.py        # Point d'entrée des parties
├── master_hex.py      # Adaptateur Seahorse
└── player_hex.py      # Classe de base des joueurs
```