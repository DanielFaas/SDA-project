# Temporary file to see how the data looks. TODO: Remove this file

import pandas as pd

games = pd.read_csv("games_cleaned.csv")
chess_games1 = pd.read_csv("chess_games_cleaned_part1.csv")
chess_games2 = pd.read_csv("chess_games_cleaned_part2.csv")
chess_games = pd.concat([chess_games1, chess_games2], ignore_index=True)


def summarize_dataset(df, name):
    print("\n" + "="*70)
    print(f"SUMMARY FOR: {name}")
    print("="*70)

    # Numeric summary
    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe(include="number"))

    # Categorical summary
    categorical_cols = df.select_dtypes(include="object").columns
    print("\n--- CATEGORICAL SUMMARY (unique values + counts) ---")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False))
        print("-"*40)



summarize_dataset(games, "GAMES DATASET")

# To avoid scientific notation for the big dataset
pd.set_option('display.float_format', '{:,.0f}'.format)
summarize_dataset(chess_games, "CHESS GAMES DATASET")

