"""
Opening Risk Analysis Module

This module analyzes the relationship between chess opening risk scores and game outcomes
using multinomal logistic regression and a One-Way ANOVA test. ANOVA is used to determine
games that end in draws have a different average risk than games that end normally, while
logistic regression quantifies how opening risk influences the likelihood of decisive outcomes.
High rating players frequently play safer openings and draw more frequently, so we need to use
rating as a control variable in our regression.
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD DATA

chess_games1 = pd.read_csv("./Dataset/chess_games_risk_part1.csv")

chess_games2 = pd.read_csv("./Dataset/chess_games_risk_part2.csv")

df = pd.concat([chess_games1, chess_games2], ignore_index=True, sort=False)

# Clean whitespace from event column
df['event'] = df['event'].str.strip()

# FEATURE ENGINEERING

def categorize_outcome(row):
    if row['winner'] == 'draw':
        return 'Draw'
    elif row['termination_reason'] == 'Time forfeit':
        return 'Time Forfeit'
    else:
        return 'Normal Decisive'
    
df['game_outcome'] = df.apply(categorize_outcome, axis=1)

# DATA VISUALIZATION

plt.figure(figsize=(10, 6))
sns.boxplot(x='game_outcome', y='mean_risk', data=df, palette='Set3', hue='game_outcome')
plt.title('Distribution of Opening Risk by Game Outcome')
plt.ylabel('Opening Risk (0 - 100)')
plt.xlabel('Game Outcome')
plt.savefig('./Plots/opening_risk_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# ONE-WAY ANOVA TEST

tukey = pairwise_tukeyhsd(endog=df['mean_risk'],
                          groups=df['game_outcome'],
                          alpha=0.05)

# Save Tukey test summary to file
with open('./Plots/tukey_summary.txt', 'w') as f:
    f.write(str(tukey.summary()))

# Plotting the means with confidence intervals
group_stats = df.groupby('game_outcome')['mean_risk'].agg(['mean', 'sem']).reset_index()
group_stats['ci'] = 1.96 * group_stats['sem'] # 95% CI

plt.figure(figsize=(10, 6))
plt.bar(group_stats['game_outcome'], group_stats['mean'], 
    yerr=group_stats['ci'], capsize=10, color='steelblue', alpha=0.7, edgecolor='black')
plt.title('Mean Opening Risk by Game Outcome')
plt.ylabel('Mean Opening Risk (0 - 100)')
plt.xlabel('Game Outcome')
plt.tight_layout()
plt.savefig('./Plots/tukey_mean_risk.png', dpi=300, bbox_inches='tight')
plt.close()

# MULTINOMIAL LOGISTIC REGRESSION

# Calculate average rating per game
df['avg_rating'] = (df['white_rating'] + df['black_rating']) / 2

# Convert game_outcome to categorical with specified order
df['game_outcome'] = pd.Categorical(
    df['game_outcome'], 
    categories=['Draw', 'Normal Decisive', 'Time Forfeit'], 
    ordered=True 
)

# Create numeric code for the formula because it kept giving dumb errors
df['outcome_code'] = df['game_outcome'].cat.codes

# RUN MODEL

# Define formula using numeric target
model_formula = 'outcome_code ~ mean_risk + avg_rating + C(event)'

print("Running Multinomial Logistic Regression...")
mnlogit_model = smf.mnlogit(formula=model_formula, data=df).fit()

# Save summary to file
with open('./Plots/mnlogit_summary.txt', 'w') as f:
    f.write(str(mnlogit_model.summary()))
