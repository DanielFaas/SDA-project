# This file analyses the third research question

#TODO: Expand this comment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
chess_games1 = pd.read_csv("../Dataset/chess_games_risk_part1.csv")
chess_games2 = pd.read_csv("../Dataset/chess_games_risk_part2.csv")
full_df = pd.concat([chess_games1, chess_games2], ignore_index=True)

# Standardize text
full_df["winner"] = full_df["winner"].str.lower()
full_df["event"] = full_df["event"].str.lower()

# Run everything for both blitz games and classical games
for game_type in ["c", "b"]:
    
    game_name = 'Classical' if game_type == 'c' else 'Blitz'
    name_for_plot = 'classical' if game_type == 'c' else 'blitz'
    print("\n" + "=" * 70)
    print(f"ANALYSIS FOR {game_name} GAMES")
    print("=" * 70)
    
    df = full_df[full_df["event"] == game_type].copy()

    white_df = df[["white_rating", "black_rating", "winner", "mean_risk"]].copy()
    white_df.columns = ["player_elo", "opponent_elo", "winner", "mean_risk"]
    white_df["color"] = "white"
    white_df["win"] = (white_df["winner"] == "white").astype(int)
    
    black_df = df[["black_rating", "white_rating", "winner", "mean_risk"]].copy()
    black_df.columns = ["player_elo", "opponent_elo", "winner", "mean_risk"]
    black_df["color"] = "black"
    black_df["win"] = (black_df["winner"] == "black").astype(int)
    
    player_df = pd.concat([white_df, black_df], ignore_index=True)
    player_df.dropna(inplace=True)
    
    player_df["elo_diff"] = player_df["player_elo"] - player_df["opponent_elo"]
    player_df["is_white"] = (player_df["color"] == "white").astype(int)
    
    player_df["elo_c"] = (
        player_df["player_elo"] - player_df["player_elo"].mean()
    ) / player_df["player_elo"].std()
    
    # Create ELO categories for visualization
    player_df["elo_category"] = pd.cut(
        player_df["player_elo"],
        bins=[0, 1400, 1800, 2200, 3000],
        labels=["Novice (<1400)", "Intermediate (1400-1800)", 
                "Advanced (1800-2200)", "Expert (>2200)"]
    )
    
    # PLOT 1: Statistics of the dataset
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{game_name}:Statistics of the dataset', fontsize=16, fontweight='bold')
    
    # Distribution of risk by outcome
    axes[0, 0].hist([player_df[player_df["win"]==1]["mean_risk"],
                     player_df[player_df["win"]==0]["mean_risk"]], 
                    label=["Win", "Loss/Draw"], alpha=0.7, bins=30)
    axes[0, 0].set_xlabel("Opening Risk")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Risk Distribution by Outcome")
    axes[0, 0].legend()
    
    # ELO distribution
    axes[0, 1].hist(player_df["player_elo"], bins=30, alpha=0.7)
    axes[0, 1].set_xlabel("Player ELO")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Player ELO Distribution")
    axes[0, 1].axvline(player_df["player_elo"].mean(), color='red', 
                       linestyle='--', label=f'Mean: {player_df["player_elo"].mean():.0f}')
    axes[0, 1].legend()
    
    # Win rate by ELO category
    win_by_elo = player_df.groupby("elo_category", observed=False)["win"].mean()
    axes[1, 0].bar(range(len(win_by_elo)), win_by_elo.values, color='coral', alpha=0.7)
    axes[1, 0].set_xticks(range(len(win_by_elo)))
    axes[1, 0].set_xticklabels(win_by_elo.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel("Win Rate")
    axes[1, 0].set_title("Win Rate for different ELO Categories")
    axes[1, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Risk by ELO category
    risk_by_elo = player_df.groupby("elo_category", observed=False)["mean_risk"].mean()
    axes[1, 1].bar(range(len(risk_by_elo)), risk_by_elo.values, color='lightgreen', alpha=0.7)
    axes[1, 1].set_xticks(range(len(risk_by_elo)))
    axes[1, 1].set_xticklabels(risk_by_elo.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel("Mean Risk")
    axes[1, 1].set_title("Mean Opening Risk for different ELO Categories")
    
    plt.tight_layout()
    plt.savefig(f'../Plots/statistics_for_{name_for_plot}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # PLOT 2: Win Rate vs Risk
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'{game_name}: Win Rate vs Risk for different ELO Quartiles', fontsize=16, fontweight='bold')
    
    # Win rate vs risk by ELO quartiles
    player_df["elo_quartile"] = pd.qcut(player_df["player_elo"], q=4, 
                                         labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])
    player_df["risk_bin"] = pd.cut(player_df["mean_risk"], bins=10)
    
    for quartile in player_df["elo_quartile"].unique():
        subset = player_df[player_df["elo_quartile"] == quartile]
        win_by_risk = subset.groupby("risk_bin", observed=False)["win"].mean()
        risk_midpoints = [interval.mid for interval in win_by_risk.index]
        ax.plot(risk_midpoints, win_by_risk.values, marker='o', label=quartile, linewidth=2)
    
    ax.set_xlabel("Opening Risk")
    ax.set_ylabel("Win Rate")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../Plots/win_rate_vs_risk_for_{name_for_plot}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    # Logistic Regression
    model = smf.logit(
        formula="win ~ mean_risk * elo_c + elo_diff + is_white",
        data=player_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": player_df["player_elo"]}
    )
     
    interaction_coef = model.params["mean_risk:elo_c"]
    interaction_pval = model.pvalues["mean_risk:elo_c"]
   
    print(f"Interaction Coefficient: {interaction_coef:.4f}")
    print(f"Interaction P-value: {interaction_pval:.4f}")
    
    if interaction_pval < 0.05:
        if interaction_coef > 0:
            print("H1 Accepted: Risk benefits higher-rated players more (positive interaction)")
        else:
            print("H1: Risk benefits lower-rated players more (negative interaction)")
    else:
        print("H0 Kept: No significant difference in risk effect across skill levels")
    

    # PLOT 3: Model Predictions
    risk_range = np.linspace(player_df["mean_risk"].min(), 
                            player_df["mean_risk"].max(), 100)
    elo_levels = [-2, -1, 0, 1, 2]
    
    pred_list = []
    for elo_std in elo_levels:
        pred_df_temp = pd.DataFrame({
            "mean_risk": risk_range,
            "elo_c": elo_std,
            "elo_diff": 0,
            "is_white": 0
        })
        pred_df_temp["elo_actual"] = (elo_std * player_df["player_elo"].std() + 
                                      player_df["player_elo"].mean())
        pred_df_temp["pred_win_prob"] = model.predict(pred_df_temp)
        pred_df_temp["skill_level"] = f"ELO ≈ {pred_df_temp['elo_actual'].iloc[0]:.0f}"
        pred_list.append(pred_df_temp)
    
    pred_all = pd.concat(pred_list, ignore_index=True)
    
    plt.figure(figsize=(12, 7))
    for skill in pred_all["skill_level"].unique():
        subset = pred_all[pred_all["skill_level"] == skill]
        plt.plot(subset["mean_risk"], subset["pred_win_prob"], 
                label=skill, linewidth=2.5, alpha=0.8)
    
    plt.xlabel("Opening Risk", fontsize=12)
    plt.ylabel("Predicted Win Probability", fontsize=12)
    plt.title(f"{game_name}: Effect of Opening Risk by Player ELO", 
              fontsize=14, fontweight='bold')
    plt.legend(title="ELO", fontsize=10)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f'../Plots/risk_prediction_per_elo_for_{name_for_plot}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # PLOT 6: Heatmap of Win Rates
    player_df["risk_quintile"] = pd.qcut(player_df["mean_risk"], q=5, 
                                         labels=["Very Low", "Low", "Medium", "High", "Very High"])
    
    heatmap_data = player_df.groupby(["elo_category", "risk_quintile"], observed=False)["win"].agg(['mean', 'count']).reset_index()
    heatmap_pivot = heatmap_data.pivot(index="elo_category", columns="risk_quintile", values="mean")
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
               center=0.5, vmin=0.3, vmax=0.7, cbar_kws={'label': 'Win Rate'})
    plt.title(f"{game_name}: Win Rate Heatmap by ELO and Risk Level", 
             fontsize=14, fontweight='bold')
    plt.xlabel("Risk Quintile", fontsize=12)
    plt.ylabel("ELO Category", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../Plots/heatmap_of_win_rates_for_{name_for_plot}.png', dpi=300, bbox_inches='tight')
    plt.close()
    