
# This file is used to analyse ELO rating & mean risk as predictors of game termination type
# Main model: Multinomial Logistic Regression

import pandas as pd
import numpy as np
from scipy.stats import chi2
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# PRE-PROCESSING
# Drop time forfeits 

df = pd.read_csv('Dataset/chess_games.csv')
df = df[df["termination_reason"] == "Normal"].copy()

# Create termination type categories
df["termination_type"] = "Draw"
df.loc[df["winner"] == "white", "termination_type"] = "Win"
df.loc[df["winner"] == "black", "termination_type"] = "Loss"

# Map onto numeric codes (for later analyses)
termination_map = {
    "Draw": 0,
    "Win": 1,
    "Loss": 2
}
df["termination_code"] = df["termination_type"].map(termination_map)

# Check for missing rows
assert df["termination_code"].isna().sum() == 0

# Centre predictors (for better interpretability)
df["risk_cnt"] = df["mean_risk"] - df["mean_risk"].mean()
df["elo_cnt"] = df["white_rating"] - df["white_rating"].mean()

#  DESCRIPTIVE STATISTICS 

print(df["event"].value_counts())
print(df["termination_type"].value_counts(normalize=True))
print(df[["mean_risk", "white_rating"]].describe())

# MLR CLASSIC 
# Predictiors: mean_risk, white_rating 
# Outcome: termination_code (Draw/Win/Loss)
df_c = df[df["event"] == "c"]

# Main Effects of ELO & Risk on Termination Type (no relationship)
MLR_main_c = smf.mnlogit('termination_code ~ risk_cnt + elo_cnt', 
                         data=df_c).fit(maxiter=100)


# Interaction Effect of ELO x Risk on Termination Type (interaction)
MLR_int_c = smf.mnlogit('termination_code ~ risk_cnt * elo_cnt',
                        data = df_c).fit(maxiter=100)


print('Interaction Effects MLR Regression for Classical Games')
print(MLR_int_c.summary())

# Likelihood Ratio Test (for model comparison)
def LR_test(MLR_main, MLR_interaction):
    
    ll_1 = MLR_main.llf
    ll_2 = MLR_interaction.llf

    df_1 = MLR_main.df_model
    df_2 = MLR_interaction.df_model
    df_diff = df_2 - df_1

    # likelihood ratio
    LR = 2 * (ll_2 - ll_1)
    p_value = chi2.sf(LR, df_diff)

    return LR, df_diff, p_value

LR_c, diff_c, p_c = LR_test(MLR_main_c, MLR_int_c)
print(f"Likelihood Ratio Test: LR = {LR_c}, df = {diff_c}, p = {p_c}")

# MLR BLITZ
df_b = df[df["event"] == "b"]

# Main Effects of ELO & Risk on Termination Type (No relationship)
MLR_main_b = smf.mnlogit(
    "termination_code ~ risk_cnt + elo_cnt",
    data=df_b
).fit(maxiter=100)


# Interaction Effect of ELO & Risk on Termination Type (Moderation)
MLR_int_b = smf.mnlogit(
    "termination_code ~ risk_cnt * elo_cnt",
    data=df_b
).fit(maxiter=100)

print('Interaction Effects MLR Regression for Blitz Games')
print(MLR_int_b.summary())


LR_b = 2 * (MLR_int_b.llf - MLR_main_b.llf)
df_diff_b = MLR_int_b.df_model - MLR_main_b.df_model
p_b = chi2.sf(LR_b, df_diff_b)

LR_b, df_diff_b, p_b = LR_test(MLR_main_b, MLR_int_b)
print(f"Likelihood Ratio Test: LR = {LR_b}, df = {df_diff_b}, p = {p_b}")

# DATA VISUALISATION

# ELO levels (quartiles)
elo_levels_c = np.percentile(df_c["white_rating"], [25, 50, 75]).astype(int)
elo_levels_b = np.percentile(df_b["white_rating"], [25, 50, 75]).astype(int)

risk_grid = np.linspace(0, 100, 200)
risk_mean = df["mean_risk"].mean()
elo_mean = df["white_rating"].mean()

#CLASSICAL GAMES

# 1) Win vs Draw log-Odds across Opening Risk by ELO 
plt.figure(figsize=(8, 5))

colors = ["#8B4A8E", "#42CE62", "#6279BF"]
elo_levels = np.percentile(df_c["white_rating"], [25, 50, 75]).astype(int)

for i, (elo, col) in enumerate(zip(elo_levels, colors)):
    grid = pd.DataFrame({
        "risk_cnt": risk_grid - risk_mean,
        "elo_cnt":  elo - elo_mean
    })

    probs = MLR_int_c.predict(grid)
    probs = pd.DataFrame(probs, columns=[0, 1, 2])

    # Compute log-Odds ratios (better interpretability than coefficients)
    win_draw = np.log(probs[1] / (probs[0] + 1e-12))
    label = ["Q1 (Low ELO)", "Q2 (Avg ELO)", "Q3 (High ELO)"][i]

    plt.plot(risk_grid, win_draw, color=col, linewidth=2.5, label=label)

plt.xlabel("Opening risk")
plt.ylabel("Log odds ratio (Win / Draw)")
plt.title('Win vs Draw log-Odds across Opening Risk by ELO ')
plt.xlim(0, 100)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("Plots/classical_win_vs_draw.png")
plt.show()

# 2) Loss vs Draw log-Odds across Opening Risk by ELO 
plt.figure(figsize=(8, 5))

for i, (elo, col) in enumerate(zip(elo_levels, colors)):
    grid = pd.DataFrame({
        "risk_cnt": risk_grid - risk_mean,
        "elo_cnt":  elo - elo_mean
    })

    probs = MLR_int_c.predict(grid)
    probs = pd.DataFrame(probs, columns=[0, 1, 2])

    loss_draw = np.log(probs[2] / (probs[0] + 1e-12))
    label = ["Q1 (Low ELO)", "Q2 (Avg ELO)", "Q3 (High ELO)"][i]

    plt.plot(risk_grid, loss_draw, color=col, linewidth=2.5, label=label)

plt.xlabel("Opening risk")
plt.ylabel("Log odds ratio (Loss / Draw)")
plt.xlim(0, 100)
plt.grid(alpha=0.2)
plt.legend()
plt.title("Loss vs Draw log-Odds across Opening Risk by ELO")
plt.savefig("Plots/classical_loss_vs_draw.png")
plt.show()

# BLITZ

# 3) Win vs Draw log-Odds across Opening Risk by ELO
plt.figure(figsize=(8, 5))
colors2 = ["#F5DA0E","#49F2F5","#EF4040"]

for i, (elo, col) in enumerate(zip(elo_levels, colors2)):
    grid = pd.DataFrame({
        "risk_cnt": risk_grid - risk_mean,
        "elo_cnt":  elo - elo_mean
    })

    probs = MLR_int_b.predict(grid)
    probs = pd.DataFrame(probs, columns=[0, 1, 2])

    win_draw = np.log(probs[1] / (probs[0] + 1e-12))
    label = ["Q1 (Low ELO)", "Q2 (Avg ELO)", "Q3 (High ELO)"][i]

    plt.plot(risk_grid, win_draw, color=col, linewidth=2.5, label=label)

plt.xlabel("Opening risk")
plt.ylabel("Log odds ratio (Win / Draw)")
plt.xlim(0, 100)
plt.grid(alpha=0.2)
plt.legend()
plt.title("Win vs Draw log-Odds across Opening Risk by ELO")
plt.savefig("Plots/blitz_win_vs_draw.png")
plt.show()

# 4) Loss vs Draw log-Odds across Opening Risk by ELO
plt.figure(figsize=(8, 5))

for i, (elo, col) in enumerate(zip(elo_levels, colors2)):
    grid = pd.DataFrame({
        "risk_cnt": risk_grid - risk_mean,
        "elo_cnt":  elo - elo_mean
    })

    probs = MLR_int_b.predict(grid)
    probs = pd.DataFrame(probs, columns=[0, 1, 2])

    loss_draw = np.log(probs[2] / (probs[0] + 1e-12))
    label = ["Q1 (Low ELO)", "Q2 (Avg ELO)", "Q3 (High ELO)"][i]

    plt.plot(risk_grid, loss_draw, color=col, linewidth=2.5, label=label)

plt.xlabel("Opening risk")
plt.ylabel("Log odds ratio (Loss / Draw)")
plt.xlim(0, 100)
plt.grid(alpha=0.2)
plt.legend()
plt.title("Loss vs Draw log-Odds across Opening Risk by ELO")
plt.savefig("Plots/blitz_loss_vs_draw.png")
plt.show()