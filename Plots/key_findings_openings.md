# Key Findings: Opening Risk Analysis

## Opening Risk Predicts Draw Probability
Our analysis confirms that **higher opening risk scores are statistically significantly associated with a lower probability of a game ending in a draw**, even after controlling for player rating and time control.

### Logistic Regression Results (Draw vs. Normal Decisive)
We modeled the probability of a draw (1) versus a decisive result (0).
*   **Blitz**: Opening risk coefficient = **-0.0322** ($p < 0.001$)
*   **Classical**: Opening risk coefficient = **-0.0294** ($p < 0.001$)

**Interpretation**:
*   The negative coefficients indicate that as opening risk increases, the likelihood of a draw decreases.
*   **Marginal Effects**: A one standard deviation increase in opening risk reduces the probability of a draw by approximately **0.17%** in Blitz and **0.14%** in Classical games. While the absolute magnitude is small (due to the low base rate of draws, ~5%), the effect is consistent and highly significant.

## Descriptive Statistics
Raw data shows a consistent pattern where drawn games feature safer openings on average than decisive games.

| Time Control | Outcome | Mean Opening Risk |
| :--- | :--- | :--- |
| **Blitz** | Draw | **25.97** |
| | Normal Decisive | **27.14** |
| **Classical** | Draw | **28.28** |
| | Normal Decisive | **29.26** |

*   **Hypothesis Testing**: A Kruskal-Wallis test confirms these differences are statistically significant for both time controls ($p < 0.001$).

## Control Variables
*   **Player Rating**: As expected, average player rating is a very strong positive predictor of draws ($z > 30$). Higher-rated players are much more likely to draw, but the effect of opening risk persists independently of this factor.

## Time Forfeit Analysis
We analyzed time forfeits separately to justify their exclusion from the main model.

*   **Opening Risk is Irrelevant**: In a logistic regression predicting time forfeits, the coefficient for opening risk was **0.0008** ($p = 0.651$), indicating **no statistical relationship** between opening risk and running out of time.
*   **Drivers of Forfeits**: Time forfeits are primarily driven by the time control itself (Blitz > Classical) and rating differences, not the opening choice.
*   **Robustness**: Including time forfeits in the main model (treating them as losses) barely changed the main risk coefficient (from **-0.0294** to **-0.0302**), confirming our results are robust.

## Methodological Rationale

### Why Binary Logistic Regression?
We chose **Binary Logistic Regression** (Draw vs. Normal Decisive) as our primary model instead of Linear Regression or Multinomial Logistic Regression for several reasons:
1.  **Nature of Outcome**: Our dependent variable is categorical (Draw vs. Win/Loss), making Linear Regression inappropriate as it violates assumptions of normality and can predict probabilities outside the 0-1 range.
2.  **Focus on Game Quality**: By excluding time forfeits (which we proved are unrelated to opening risk), we isolated the "chess quality" outcome. A binary model provides the cleanest interpretation of how risk affects the specific likelihood of a draw.
3.  **Interpretability**: Logistic regression coefficients can be easily converted to Odds Ratios and Marginal Effects, providing clear insights (e.g., "% change in draw probability").

### Why Kruskal-Wallis instead of ANOVA?
We utilized the **Kruskal-Wallis H-test** to compare risk distributions because:
1.  **Non-Normality**: Chess data often follows non-normal distributions. The Kruskal-Wallis test is a non-parametric alternative to One-Way ANOVA that does not assume the data is normally distributed.
2.  **Robustness**: It is more robust to outliers, which are common in large datasets of chess games.

### Why Separate Models for Time Controls?
We ran separate models for **Blitz** and **Classical** rather than pooling them because:
1.  **Distinct Dynamics**: The strategic nature of chess changes fundamentally under time pressure. Blunders are more common in Blitz, while Classical games are deeper.
2.  **Avoid Interaction Complexity**: While a pooled model with interaction terms (`Risk * TimeControl`) is possible, separate models allow for clearer, direct comparison of effect sizes and significance levels for each specific domain.

### Why Control for Rating?
We included **Average Rating** as a control variable because it is a known confounder:
*   Higher-rated players are significantly more likely to draw.
*   Higher-rated players may also select different openings.
*   Without controlling for rating, we could not be sure if the observed effect was due to the opening itself or simply the skill level of the players choosing those openings.
