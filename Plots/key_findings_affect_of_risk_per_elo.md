## Key Findings: Opening Risk and Winning Probability Conditional on Player Skill

### Opening Risk Is Associated with Winning Probability, Conditional on Elo

Our analysis shows that the relationship between opening risk and winning probability depends on player skill. After controlling for player Elo, opponent Elo, and color, opening risk is systematically associated with changes in a player’s probability of winning, and this association varies across skill levels.

Rather than having a uniform effect, opening risk interacts with player Elo: the same level of risk does not affect lower- and higher-rated players in the same way.

---

### Logistic Regression Results (Win vs. Loss/Draw)

We modeled the probability of a player winning a game (1) versus not winning (0, including losses and draws) using binary logistic regression.


**Key results:**

- **Main Effect of Opening Risk:** Opening risk is significantly associated with winning probability.
- **Risk × Elo Interaction:** The interaction between opening risk and player Elo is statistically significant, indicating that the effect of opening risk depends on player skill.

---

### Interpretation

- **Skill-Dependent Risk Effects:**  
  Higher-risk openings do not affect all players equally. Stronger players are better able to manage or exploit risky openings, whereas weaker players are more likely to suffer from the increased volatility associated with such openings.

- **Probabilistic, Not Deterministic:**  
  Opening risk shifts the probability of winning; it does not determine outcomes. Riskier openings increase variance in outcomes rather than guaranteeing success or failure.

---

### Descriptive Patterns

Descriptive analyses support the regression results:

- Winning probability varies with opening risk across Elo groups.
- Higher-rated players show more favorable win-rate curves as risk increases.
- Lower-rated players experience a decline in win probability when choosing high-risk openings.

These patterns are consistent across both Blitz and Classical games, though the magnitude of effects differs by time control.

---

### Visualization-Based Evidence

- **Win Rate vs. Risk Plots:**  
  Stratifying players into Elo quartiles reveals different win-rate trends across risk levels, visually confirming the interaction between risk and skill.

- **Heatmaps:**  
  Heatmaps of win rates by Elo category and risk quintile show that win rates differ for both different ELO categories and opening risk.

- **Model-Based Predictions:**  
  Predicted probabilities from the logistic regression illustrate that, at identical risk levels, higher-Elo players maintain higher chances of winning.

---

### Methodological Rationale

#### Why Binary Logistic Regression?

Binary logistic regression is appropriate because:

- **Outcome Structure:** The dependent variable (win vs. loss/draw) is binary.
- **Interpretability:** Coefficients can be interpreted via odds ratios and marginal effects.
- **Interaction Testing:** The model allows direct testing of whether opening risk effects vary by player skill.

Linear regression would violate key assumptions, and multinomial models are unnecessary given the research focus on winning probability.

---

### Why Treat Elo as a Continuous Variable?

- Elo is inherently continuous and represents differences in player skill.
- Using continuous Elo preserves statistical power and avoids cutoffs.
- Interaction effects between risk and skill are more meaningfully interpreted on a continuous scale.
- Elo categories are used only for visualization, not inference.

---

### Why Separate Models for Blitz and Classical?

We estimated separate models because:

- **Distinct Playing Conditions:** Time pressure fundamentally alters decision-making and error rates.
- **Interpretability:** Separate models allow direct comparison of effect sizes without higher-order interaction terms.
- **Robustness:** Results are consistent across time controls.

---

### Limitations

- Opening risk isn't uniformly distributed across games, high-risk openings occur less often than lower-risk openings
- Opening choiche is not random, especially for more experianced players.
- The analysis focusses on early-game decisions, while late-game decisions also affect the outcome of the game
