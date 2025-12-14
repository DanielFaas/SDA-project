# This file is used to cleanup the two datasets used for this project

# TODO: add script for downloading the original dataset
# TODO: Fix the problem with blitz vs classical
import pandas as pd
import numpy as np
import re

def extract_moves(an_string):
    s = re.sub(r"\{[^}]*\}|\([^)]+\)|\d-\d|1-0|0-1|1/2-1/2", "", an_string)
    s = re.sub(r"\d+\.(\.\.)?", "", s)
    moves = s.split()
    return moves


chess_games = pd.read_csv("chess_games.csv")


chess_games = chess_games.drop_duplicates(subset=['White', 'Black', 'UTCDate', 'UTCTime', 'WhiteElo', 'BlackElo'])

chess_games = chess_games[
    (
        chess_games['Event'].str.contains('Blitz', case=False, na=False)
    )
    &
    ~chess_games['Event'].str.contains('tournament', case=False, na=False)
]


chess_games = chess_games[
    ~chess_games['Termination'].str.contains('abandoned', case=False, na=False) &
    ~chess_games['Termination'].str.contains('rules', case=False, na=False)
]

chess_games['AN'] = chess_games['AN'].apply(extract_moves)


# Cut content: White, Black, UTCDate, UTCTime, WhiteRatingDiff, BlackRatingDiff, AN, Opening_eco

chess_games_small = chess_games[['Event', 'WhiteElo', 'BlackElo', 'Result', 'Opening', 'Termination', 'AN']]

chess_games_small = chess_games_small.rename(columns={
    'Event': 'event',
    'WhiteElo': 'white_rating',
    'BlackElo': 'black_rating',
    'Result': 'winner',
    'Opening': 'opening_name',
    'Termination': 'termination_reason'
})

result_map = {'1-0': 'white', '0-1': 'black', '1/2-1/2': 'draw'}
chess_games_small['winner'] = chess_games_small['winner'].map(result_map).fillna('unknown')

# chess_games_small = fix_text_columns(chess_games_small)
chess_games_small['white_rating'] = pd.to_numeric(chess_games_small['white_rating'], errors='coerce').astype('Int32')
chess_games_small['black_rating'] = pd.to_numeric(chess_games_small['black_rating'], errors='coerce').astype('Int32')

print(chess_games_small)

chess_games_small = chess_games_small.iloc[:1000000]

# chess_games_small.to_csv("chess_games_cleaned_blitz.csv", index=False)
