"""
Opening Risk Analysis Module

This module implements a comprehensive analysis of chess opening risk and game outcomes.
Split by time control (Blitz vs Classical)
Focus on Draw vs Normal Decisive (excluding time forfeits)
Use Binary Logistic Regression to model Draw probability
Visualize results with plots.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import kruskal
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set_style("whitegrid")

# DATA LOADING AND PREPARATION
print("Loading data...")
chess_games1 = pd.read_csv("./Dataset/chess_games_risk_part1.csv")
chess_games2 = pd.read_csv("./Dataset/chess_games_risk_part2.csv")
df = pd.concat([chess_games1, chess_games2], ignore_index=True, sort=False)

# Map event to time_control
# where b = Blitz, c = Classical
event_map = {'b': 'Blitz', 'c': 'Classical'}
df['time_control'] = df['event'].map(event_map)

# Categorize outcome
def categorize_outcome(row):
    if row['winner'] == 'draw':
        return 'Draw'
    elif row['termination_reason'] == 'Time forfeit':
        return 'Time Forfeit'
    else:
        return 'Normal Decisive'

df['game_outcome'] = df.apply(categorize_outcome, axis=1)

# Calculate average rating
df['avg_rating'] = (df['white_rating'] + df['black_rating']) / 2

# Exclude Time Forfeits for main analysis
df_played = df[df['game_outcome'] != 'Time Forfeit'].copy()

# Create Binary Outcome: Draw = 1, Normal Decisive = 0
df_played['draw_binary'] = (df_played['game_outcome'] == 'Draw').astype(int)

# Standardize predictors (Z-scores) for better interpretability in regression
# This helps in comparing effect sizes of Risk vs Rating
df_played['mean_risk_z'] = (df_played['mean_risk'] - df_played['mean_risk'].mean()) / df_played['mean_risk'].std()
df_played['avg_rating_z'] = (df_played['avg_rating'] - df_played['avg_rating'].mean()) / df_played['avg_rating'].std()

print(f"Data loaded. Total games: {len(df)}. Played games (excluding time forfeits): {len(df_played)}")

# DESCRIPTIVE STATISTICS

# Summary by Time Control and Outcome
summary = (
    df_played
    .groupby(['time_control', 'game_outcome'])
    .agg(
        mean_risk=('mean_risk', 'mean'),
        sd_risk=('mean_risk', 'std'),
        n=('mean_risk', 'count')
    )
)
print("\n--- Risk Stats by Outcome & Time Control ---")
print(summary)

# Overall Draw Rates by Time Control
summary_tc = (
    df_played
    .groupby(['time_control'])
    .agg(
        draw_rate=('draw_binary', 'mean'),
        mean_risk=('mean_risk', 'mean'),
        n=('mean_risk', 'count')
    )
)
print("\n--- Overall Stats by Time Control ---")
print(summary_tc)


# HYPOTHESIS TESTING (Risk Differences)

for tc in ['Blitz', 'Classical']:
    subset = df_played[df_played['time_control'] == tc]
    if subset['draw_binary'].nunique() > 1:
        stat, p = kruskal(
            subset[subset['game_outcome']=='Draw']['mean_risk'],
            subset[subset['game_outcome']=='Normal Decisive']['mean_risk']
        )
        print(f"{tc}: Statistic={stat:.4f}, p-value={p:.4e}")
    else:
        print(f"{tc}: Not enough data for comparison")

# LOGISTIC REGRESSION MODELS

models = {}

for tc in ['Blitz', 'Classical']:
    print(f"\nRunning Model for: {tc}")
    subset = df_played[df_played['time_control'] == tc].copy()
    
    # Check if we have enough variation in event
    if subset['event'].nunique() > 1:
        formula = 'draw_binary ~ mean_risk_z + avg_rating_z + C(event)'
    else:
        formula = 'draw_binary ~ mean_risk_z + avg_rating_z'
        
    try:
        logit_model = smf.logit(formula=formula, data=subset).fit(disp=0)
        models[tc] = logit_model
        print(logit_model.summary())
        
        # Marginal Effects
        try:
            mfx = logit_model.get_margeff()
            print(f"\nMarginal Effects for {tc}:")
            print(mfx.summary())
        except Exception as e:
            print(f"Could not calculate marginal effects: {e}")
        
    except Exception as e:
        print(f"Error running model for {tc}: {e}")


# VISUALIZATIONS

# Figure 1: Risk distributions
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_played,
    x='game_outcome',
    y='mean_risk',
    hue='time_control',
    palette='Set2'
)
plt.title('Distribution of Opening Risk by Game Outcome and Time Control')
plt.ylabel('Opening Risk')
plt.xlabel('Game Outcome')
plt.savefig('./Plots/opening_risk_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 1: Risk distributions")

# Figure 2: Draw probability vs opening risk (binned)
# Use qcut with duplicates='drop' to handle potential duplicate bin edges
df_played['risk_bin'] = pd.qcut(df_played['mean_risk'], 10, duplicates='drop')

plot_df = (
    df_played
    .groupby(['time_control', 'risk_bin'], observed=False)
    .agg(
        draw_rate=('draw_binary', 'mean'),
        mean_risk_bin=('mean_risk', 'mean')
    )
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df, x='mean_risk_bin', y='draw_rate', hue='time_control', marker='o', palette='Set1')
plt.title('Draw Rate vs Opening Risk (Deciles)')
plt.xlabel('Mean Opening Risk (Binned)')
plt.ylabel('Draw Rate')
plt.grid(True, alpha=0.3)
plt.savefig('./Plots/opening_draw_rate_binned.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 2: Draw rate vs Risk (Binned)")

# Figure 3: Model-based predicted probabilities
plt.figure(figsize=(10, 6))

# Create a range of risk values (Z-scores)
risk_range_z = np.linspace(df_played['mean_risk_z'].min(), df_played['mean_risk_z'].max(), 100)
# Convert back to raw risk for plotting x-axis
risk_mean = df_played['mean_risk'].mean()
risk_std = df_played['mean_risk'].std()
risk_range_raw = risk_range_z * risk_std + risk_mean

colors = {'Blitz': 'blue', 'Classical': 'orange'}

sns.scatterplot(
    data=plot_df, 
    x='mean_risk_bin', 
    y='draw_rate', 
    hue='time_control', 
    palette=colors, 
    s=100, 
    alpha=0.6, 
    legend=False 
)

for tc, model in models.items():
    # Calculate mean rating for this specific time controlt
    subset_tc = df_played[df_played['time_control'] == tc]
    mean_rating_z_tc = subset_tc['avg_rating_z'].mean()

    # Create prediction data
    pred_data = pd.DataFrame({
        'mean_risk_z': risk_range_z,
        'avg_rating_z': mean_rating_z_tc
    })
    
    # Handle event variable if it was in the model
    if 'event' in subset_tc.columns and subset_tc['event'].nunique() > 1:
        # Use the most common event as the reference
        most_common_event = subset_tc['event'].mode()[0]
        pred_data['event'] = most_common_event
    
    try:
        probs = model.predict(pred_data)
        plt.plot(risk_range_raw, probs, label=f'{tc} (Predicted)', color=colors.get(tc, 'black'), linewidth=2)
    except Exception as e:
        print(f"Could not plot predictions for {tc}: {e}")

plt.title('Predicted Probability vs Actual Draw Rates')
plt.xlabel('Opening Risk')
plt.ylabel('Probability of Draw')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('./Plots/opening_predicted_probabilities.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 3: Predicted Probabilities")

# TIME FORFEIT ANALYSIS

# Data Prep for Forfeits
# Create binary outcome: 1 = Time Forfeit, 0 = Any other result
df['time_forfeit'] = (df['game_outcome'] == 'Time Forfeit').astype(int)

# Calculate Rating Diff
df['rating_diff'] = (df['white_rating'] - df['black_rating']).abs()

# Standardize predictors for the full dataset
df['mean_risk_z'] = (df['mean_risk'] - df['mean_risk'].mean()) / df['mean_risk'].std()
df['avg_rating_z'] = (df['avg_rating'] - df['avg_rating'].mean()) / df['avg_rating'].std()
df['rating_diff_z'] = (df['rating_diff'] - df['rating_diff'].mean()) / df['rating_diff'].std()

# Logistic Regression for Forfeits
# Model: time_forfeit ~ time_control + avg_rating + rating_diff + mean_risk

print("\nRunning Logistic Regression for Time Forfeits...")
forfeit_formula = 'time_forfeit ~ C(time_control) + avg_rating_z + rating_diff_z + mean_risk_z'

try:
    forfeit_model = smf.logit(formula=forfeit_formula, data=df).fit(disp=0)
    print(forfeit_model.summary())
    
    # Check marginal effects
    try:
        mfx_forfeit = forfeit_model.get_margeff()
        print("\nMarginal Effects for Time Forfeits:")
        print(mfx_forfeit.summary())
    except Exception as e:
        print(f"Could not calculate marginal effects for forfeits: {e}")
    
except Exception as e:
    print(f"Error running forfeit model: {e}")

# FIGURES FOR FORFEIT ANALYSIS

# Figure T1: Time forfeit rate by time control
plt.figure(figsize=(8, 6))
forfeit_rates = df.groupby('time_control')['time_forfeit'].mean().reset_index()
sns.barplot(data=forfeit_rates, x='time_control', y='time_forfeit', palette='Reds')
plt.title('Time Forfeit Rate by Time Control')
plt.ylabel('Proportion of Games Ending in Time Forfeit')
plt.xlabel('Time Control')
plt.savefig('./Plots/opening_forfeit_rate_tc.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure T1: Forfeit rate by time control")

# Figure T2: Forfeit rate vs Average Rating
# Bin average rating into deciles
df['rating_bin'] = pd.qcut(df['avg_rating'], 10)
plot_df_forfeit = df.groupby('rating_bin', observed=False)['time_forfeit'].mean().reset_index()
# Convert intervals to string for plotting
plot_df_forfeit['rating_bin_str'] = plot_df_forfeit['rating_bin'].astype(str)

plt.figure(figsize=(12, 6))
sns.barplot(data=plot_df_forfeit, x='rating_bin_str', y='time_forfeit', color='salmon')
plt.title('Time Forfeit Rate by Average Rating Decile')
plt.ylabel('Proportion of Games Ending in Time Forfeit')
plt.xlabel('Average Rating Range')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./Plots/opening_forfeit_rate_rating.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure T2: Forfeit rate by rating")

# Figure T3: Forfeit rate vs Opening Risk
print("Generating Forfeit Rate vs Risk Plot...")
df['risk_decile'] = pd.qcut(df['mean_risk'], 10, duplicates='drop')

plt.figure(figsize=(12, 6))
# Using pointplot to show the mean rate and confidence intervals
sns.pointplot(
    data=df, 
    x='risk_decile', 
    y='time_forfeit', 
    color='dimgrey',
    capsize=.1,
    linestyles='-',
    errorbar=('ci', 95)
)
plt.title('Time Forfeit Rate by Opening Risk Decile')
plt.ylabel('Proportion of Games Ending in Time Forfeit')
plt.xlabel('Opening Risk Range (Deciles)')
plt.xticks(rotation=45)

# Set Y-axis to 0-0.5 to show the lack of trend in context
plt.ylim(0, 0.5)

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./Plots/opening_forfeit_rate_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure T3: Forfeit rate by risk")

# Main Model including Forfeits
print("Running Main Model (Classical) treating Time Forfeits as Decisive Losses...")

# Create dataset where Time Forfeit is treated as Normal Decisive (0)
# Draw = 1, (Normal Decisive OR Time Forfeit) = 0
df_robust = df.copy()
df_robust['draw_binary'] = (df_robust['game_outcome'] == 'Draw').astype(int)

# Filter for Classical only for comparison
subset_robust = df_robust[df_robust['time_control'] == 'Classical']

robust_formula = 'draw_binary ~ mean_risk_z + avg_rating_z'
try:
    robust_model = smf.logit(formula=robust_formula, data=subset_robust).fit(disp=0)
    print(robust_model.summary())
    
    print("\nComparison of Risk Coefficient (Classical):")
    if 'Classical' in models:
        print(f"Original Model (Excluding Forfeits): {models['Classical'].params['mean_risk_z']:.4f}")
    print(f"Robust Model (Including Forfeits):   {robust_model.params['mean_risk_z']:.4f}")
    
except Exception as e:
    print(f"Error running robustness check: {e}")

print("\nAnalysis complete.")