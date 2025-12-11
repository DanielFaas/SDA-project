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

# We use the formula API
# Control for ratings and event type to isolate the effect of mean risk

# We set the reference category to draw so the model calculates
# Normal Decisive vs Draw and Time Forfeit vs Draw

model_formula = 'game_outcome ~ mean_risk + white_rating + black_rating + C(event)'
mnlogit_model = smf.mnlogit(formula=model_formula, data=df).fit()

print(mnlogit_model.summary())

# CALCULATE ODDS RATIOS

params = mnlogit_model.params
conf = mnlogit_model.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'Odds_Ratio']
odds_ratios = np.exp(conf)

print("\nOdds Ratios with 95% Confidence Intervals:\n", odds_ratios)

# VISUALIZE ODDS RATIOS
# Forest plot which allows us to see the magnitude and direction of effects
# on the likelihood of different outcomes relative to the reference category.

# Reset index to make the outcome categories accessible for plotting
odds_ratios_plot = odds_ratios.reset_index()
odds_ratios_plot = odds_ratios_plot.rename(columns={'level_0': 'Outcome', 'level_1': 'Variable'})

# Filter out the Intercept as it's usually on a different scale and less interpretable
odds_ratios_plot = odds_ratios_plot[odds_ratios_plot['Variable'] != 'Intercept']

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create the point plot with error bars
# We iterate through unique outcomes to plot them
outcomes = odds_ratios_plot['Outcome'].unique()
colors = sns.color_palette("husl", len(outcomes))

for i, outcome in enumerate(outcomes):
    subset = odds_ratios_plot[odds_ratios_plot['Outcome'] == outcome]
    
    plt.errorbar(x=subset['Odds_Ratio'], 
                 y=subset['Variable'], 
                 xerr=[subset['Odds_Ratio'] - subset['2.5%'], subset['97.5%'] - subset['Odds_Ratio']],
                 fmt='o', 
                 label=outcome,
                 color=colors[i],
                 capsize=5,
                 alpha=0.7)

plt.axvline(x=1, color='red', linestyle='--', linewidth=1, label='No Effect (OR=1)')
plt.title('Odds Ratios for Game Outcomes (Reference: Draw)')
plt.xlabel('Odds Ratio (log scale)')
plt.xscale('log')
plt.legend(title='Outcome vs Draw')
plt.tight_layout()
plt.savefig('./Plots/mnlogit_odds_ratios.png', dpi=300)
plt.close()
