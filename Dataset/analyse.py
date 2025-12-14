# This file gives a very quick overview of the dataset, this was used during the cleaning process to find mistakes in the dataset.

import pandas as pd


chess_games1 = pd.read_csv("chess_games_risk_part1.csv")
chess_games2 = pd.read_csv("chess_games_risk_part2.csv")


both_games = pd.concat([chess_games1, chess_games2], ignore_index=True, sort=False)

def summarize_dataset(df):
    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe(include="number"))

    categorical_cols = df.select_dtypes(include="object").columns
    print("\n--- CATEGORICAL SUMMARY (unique values + counts) ---")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False))
        print("-"*40)



# To avoid scientific notation for the big dataset
pd.set_option('display.float_format', '{:,.0f}'.format)
summarize_dataset(both_games)

