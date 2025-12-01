# This file is used to cleanup the two datasets used for this project

# TODO: add script for downloading the original datasets
import pandas as pd
import numpy as np

def fix_text_columns(df):
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )

    return df


games = pd.read_csv("games.csv", low_memory=False)


# There are 945 duplicate entries, those are dropped
games = games.drop_duplicates(subset='id')


# Cut content: id, rated, created_at, last_move_at, increment_code, white_id, black_id, moves
games_small = games[['white_rating','black_rating','winner','victory_status',
                     'opening_name', 'opening_eco', 'opening_ply', 'turns']]


# Renamed opening_ply to opening_play_turns
games_small = games_small.rename(columns={'opening_ply': 'opening_play_turns', 'victory_status': 'termination_reason'})
games_small = fix_text_columns(games_small)
games_small['winner'] = games_small['winner'].replace({'unknown': None})
print(games_small)

# Save to a new CSV file
games_small.to_csv("games_cleaned.csv", index=False)




# Second dataset (NOTE: Very large!!!)

chess_games = pd.read_csv("chess_games.csv", low_memory=False)


chess_games = chess_games.drop_duplicates(subset=['White', 'Black', 'UTCDate', 'UTCTime', 'WhiteElo', 'BlackElo'])


# Cut content: White, Black, UTCDate, UTCTime, WhiteRatingDiff, BlackRatingDiff, AN
# TODO: Add turns from AN if needed
chess_games_small = chess_games[['Event', 'WhiteElo', 'BlackElo', 'Result', 'ECO', 'Opening', 'Termination']]

chess_games_small = chess_games_small.rename(columns={
    'Event': 'game_type',
    'WhiteElo': 'white_rating',
    'BlackElo': 'black_rating',
    'Result': 'winner',
    'ECO': 'opening_eco',
    'Opening': 'opening_name',
    'Termination': 'termination_reason'
})

result_map = {'1-0': 'white', '0-1': 'black', '1/2-1/2': 'draw'}
chess_games_small['winner'] = chess_games_small['winner'].map(result_map).fillna('unknown')

chess_games_small = fix_text_columns(chess_games_small)
chess_games_small['white_rating'] = pd.to_numeric(chess_games_small['white_rating'], errors='coerce').astype('Int32')
chess_games_small['black_rating'] = pd.to_numeric(chess_games_small['black_rating'], errors='coerce').astype('Int32')
chess_games_small['winner'] = chess_games_small['winner'].replace({'unknown': None})

print(chess_games_small)

# Split into 2 roughly equal parts
# TODO: fix FutureWarning
chunks = np.array_split(chess_games_small, 2)

# Save each chunk separately
chunks[0].to_csv("chess_games_cleaned_part1.csv", index=False)
chunks[1].to_csv("chess_games_cleaned_part2.csv", index=False)