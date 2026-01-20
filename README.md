# Football Match Prediction using Poisson Model

## 1. Introduction
This project applies the Poisson distribution to model and predict football match outcomes,
focusing on the English Premier League.

Instead of predicting a single match result, the goal is to estimate probabilities
for different scorelines and betting-related events.

## 2. Problem Definition
Given historical match data:
- Can we estimate the expected number of goals for each team?
- Can these estimates be used to compute probabilities for:
  - Home / Draw / Away
  - Over / Under goals?

## 3. Dataset
- League: English Premier League
- Seasons: (ghi mùa bạn dùng)
- Data includes:
  - Match results
  - Goals scored
  - Home and away teams

## 4. Methodology
- Goal scoring is modeled using Poisson distribution
- Separate models for:
  - Home team goals
  - Away team goals

## 5. Results
- Example match predictions
- Probability distributions
- Discussion on accuracy and limitations

## 6. Limitations
- Independence assumption
- No tactical or lineup information
- Draw outcomes are inherently hard to predict

## 7. Conclusion
Poisson models provide a strong statistical baseline for football prediction,
but should be interpreted probabilistically rather than deterministically.
