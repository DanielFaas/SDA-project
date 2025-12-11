# This file calculates the risk score of an opening in a chess games
# The method used here is based on the method used in the paper Risk-taking in adversarial games: What can 1 billion online chess games tell us
# DOI: https://doi.org/10.31234/osf.io/vgpdj


import chess
import chess.engine
import networkx as nx
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import ast

# Makes a clean list of chess moves
def parse_an_to_moves(an_string_or_list):
    if isinstance(an_string_or_list, list):
        return an_string_or_list

    if isinstance(an_string_or_list, str):
        moves = ast.literal_eval(an_string_or_list)
        if isinstance(moves, list):
            return moves

    return []

# Builds a directed graph of board posistion for the first max_depth moves
def build_position_graph(games_moves, max_depth=10):
    G = nx.DiGraph()
    for moves in tqdm(games_moves, desc="Building position graph"):
        board = chess.Board()
        depth = 0
        prev_fen = board.fen()

        if prev_fen not in G:
            G.add_node(prev_fen, count=0, eval=None)
        G.nodes[prev_fen]['count'] += 1

        for san in moves:
            if depth >= max_depth:
                break
            try:
                board.push_san(san)
            except ValueError:
                break
            fen = board.fen()
            if fen not in G:
                G.add_node(fen, count=0, eval=None)
            G.nodes[fen]['count'] += 1

            if not G.has_edge(prev_fen, fen):
                G.add_edge(prev_fen, fen, count=0)
            G.edges[prev_fen, fen]['count'] += 1

            prev_fen = fen
            depth += 1
    return G


# Computes the Stockfish evaluation for each posistion in the graph
def evaluate_positions(G, engine_path="/usr/games/stockfish", depth=10):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    for fen in tqdm(G.nodes, desc="Evaluating positions"):
        if G.nodes[fen]['eval'] is not None:
            continue
        board = chess.Board(fen)
        try:
            score = engine.analyse(board, chess.engine.Limit(depth=depth))
            cp = score['score'].white().score(mate_score=100000)
            G.nodes[fen]['eval'] = cp
        except:
            G.nodes[fen]['eval'] = 0
    engine.quit()
    return G


# Computed weighted standard deviation of possible outcomes (formula from article, see comment on top of the file)
def compute_weighted_variance(evals, counts):
    if len(evals) < 2 or sum(counts) == 0:
        return 0
    V1 = sum(counts)
    V2 = sum([c**2 for c in counts])
    weighted_mean = sum([x*w for x,w in zip(evals, counts)]) / V1
    var = (V1 / (V1**2 - V2)) * sum([w*(x - weighted_mean)**2 for x,w in zip(evals, counts)])
    return np.sqrt(var)

# Computes the risk for a single move based on oppenents responses saved in graph
def compute_move_risk(board, move_san, G):
    try:
        board.push_san(move_san)
    except ValueError:
        return 0
    fen = board.fen()
    neighbor_fens = list(G.neighbors(fen))
    evals = [G.nodes[n]['eval'] for n in neighbor_fens if G.nodes[n]['eval'] is not None]
    counts = [G.edges[fen, n]['count'] for n in neighbor_fens if G.nodes[n]['eval'] is not None]
    risk = compute_weighted_variance(evals, counts)
    board.pop()
    return risk


# Computes the risk taken in a game for every move
def compute_game_risk(moves, G, max_depth=10):
    board = chess.Board()
    risks = []
    for i, san in enumerate(moves):
        if i >= max_depth:
            break
        risk = compute_move_risk(board, san, G)
        risks.append(risk)
        try:
            board.push_san(san)
        except ValueError:
            break
    if not risks:
        return [], 0, 0
    return risks, max(risks), np.mean(risks)


# Saves graph in a file for later use
def save_graph(G, filepath="graph.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump(G, f)

# Loads the graph from file
def load_graph(filepath="graph.pkl"):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# Returns the first max_depth moves
def filter_moves_for_openings(moves, max_depth=10):
    return moves[:max_depth]

def normalize_risk(values, low_percent=1, high_percent=99):
    values = np.array(values, dtype=float)

    # Step 1: Compute robust boundaries
    lower = np.percentile(values, low_percent)
    upper = np.percentile(values, high_percent)

    # Step 2: Clip
    clipped = np.clip(values, lower, upper)

    # Step 3: Min-max normalize to [0, 100]
    normalized = 100 * (clipped - lower) / (upper - lower)

    return normalized, lower, upper



df1 = pd.read_csv("chess_games_cleaned.csv")

df1["moves"] = df1["AN"].apply(parse_an_to_moves)

df2 = pd.read_csv("chess_games_cleaned_blitz.csv")

df2["moves"] = df2["AN"].apply(parse_an_to_moves)

df = pd.concat([df1, df2], ignore_index=True, sort=False)

# TODO: Fix this
G = build_position_graph(df["moves"], max_depth=6)
print("Graph nodes:", len(G.nodes))
print("Graph edges:", len(G.edges))

save_graph(G, "position.pkl")
G = load_graph("position.pkl")

G = evaluate_positions(G, engine_path="/usr/games/stockfish", depth=10)
save_graph(G, "evaluation.pkl")
G = load_graph("evaluation.pkl")


all_risks = []
all_max_risks = []
all_mean_risks = []

for game in tqdm(df["moves"], desc="Computing risk for each game"):
    _, _, mean_risk = compute_game_risk(game, G, max_depth=6)
    all_mean_risks.append(mean_risk)

all_mean_risks = [float(x) for x in all_mean_risks]

df["mean_risk"] = all_mean_risks

df = df[['event', 'white_rating', 'black_rating', 'winner', 'opening_name', 'termination_reason', 'mean_risk']]


df.to_csv("chess_games_risk.csv", index=False)

# Split into 2 roughly equal parts
df = df[df["mean_risk"] < 100]
df = df[df["mean_risk"] > 1]


df["mean_risk"], lower, upper = normalize_risk(df["mean_risk"])
print(df)

df["event"] = df["event"].str.replace(
    "Blitz", "b", case=False, regex=False
)
df["event"] = df["event"].str.replace(
    "Classical", "c", case=False, regex=False
)


midpoint = len(df) // 2

chunk1 = df.iloc[:midpoint]
chunk2 = df.iloc[midpoint:]

# # Save chunks
chunk1.to_csv("chess_games_risk_part1.csv", index=False)
chunk2.to_csv("chess_games_risk_part2.csv", index=False)
