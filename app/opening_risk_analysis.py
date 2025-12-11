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
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD DATA

df = pd.read_csv('./Data/opening_risk_data.csv')

# FEATURE ENGINEERING

def categorize_outcome(row):
    if row['winner'] == 'draw':
        return 'Draw'
    elif row['termination_reason'] == 'time forfeit':
        return 'Time Forfeit'
    else:
        return 'Normal Decisive'
    
df['game_outcome'] = df.apply(categorize_outcome, axis=1)

# DATA VISUALIZATION

plt.figure(figsize=(10, 6))
sns.boxplot(x='game_outcome', y='mean_risk', data=df, palette='Set3')
plt.title('Distribution of Opening Risk by Game Outcome')
plt.ylabel('Opening Risk (0 - 100)')
plt.xlabel('Game Outcome')
plt.savefig('./Plots/opening_risk_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# ONE-WAY ANOVA TEST

tukey = pairwise_tukeyhsd(endog=df['mean_risk'],
                          groups=df['game_outcome'],
                          alpha=0.05)

print(tukey.summary())

fig = tukey.plot_simultaneous(xlabel="Mean Risk", ylabel="Game Outcome")
plt.title("Multiple Comparison of Mean Risk by Outcome")
plt.show()

# MULTINOMIAL LOGISTIC REGRESSION


