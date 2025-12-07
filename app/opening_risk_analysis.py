"""
Opening Risk Analysis Module

This module analyzes the relationship between chess opening risk scores and game outcomes
using logistic regression. It generates synthetic chess game data with player ratings and
opening risk metrics, then fits a logistic regression model to determine whether opening
risk scores significantly predict decisive game outcomes.

Key findings are printed including model summary, odds ratios, and hypothesis test results
for the opening risk coefficient.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


# FAKE DATA GENERATION

np.random.seed(42)
n_games = 2000

data = {
    'WhiteElo': np.random.normal(1500, 200, n_games),
    'BlackElo': np.random.normal(1500, 200, n_games),
    'OpeningRiskScore': np.random.uniform(0, 10, n_games), 
}

df = pd.DataFrame(data)

def simulate_outcome(row):
    elo_diff = abs(row['WhiteElo'] - row['BlackElo'])
    prob_decisive = 0.40 + (elo_diff * 0.0005) + (row['OpeningRiskScore'] * 0.03)
    prob_decisive = min(prob_decisive, 0.95)
    return 1 if np.random.random() < prob_decisive else 0

df['IsDecisive'] = df.apply(simulate_outcome, axis=1)


# FEATURES

df['RatingDiff'] = (df['WhiteElo'] - df['BlackElo']).abs()
df['AvgRating'] = (df['WhiteElo'] + df['BlackElo']) / 2

# LOGISTIC REGRESSION MODEL

Y = df['IsDecisive']
X = df[['OpeningRiskScore', 'RatingDiff', 'AvgRating']]
X = sm.add_constant(X)
model = sm.Logit(Y, X).fit()

# RESULTS

print(model.summary())

print("\n--- ODDS RATIOS ---")
params = model.params
conf = model.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))

p_value_risk = model.pvalues['OpeningRiskScore']
coef_risk = model.params['OpeningRiskScore']

print("\n--- HYPOTHESIS TEST RESULT ---")
print(f"P-value for Opening Risk: {p_value_risk:.5f}")
