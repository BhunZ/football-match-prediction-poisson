# ⚽ Football Match Prediction (EPL) using Poisson Model

## 📌 Overview
This project explores the use of the **Poisson distribution** to model and predict football match outcomes,
with a focus on the **English Premier League (EPL)**.

Rather than predicting a single deterministic result, the objective is to **estimate probability distributions**
over possible scorelines and derive outcome-related probabilities commonly used in football analytics.

---

## 🎯 Problem Statement
Given historical EPL data, we aim to:
1. Estimate **expected goals** for each team in a match:  
   - λ_home (expected home goals)  
   - λ_away (expected away goals)
2. Use these λ values to compute:
   - **P(Home win), P(Draw), P(Away win)**
   - **P(Over/Under)** a goal line

This is a **probability estimation** task (decision-support), not a deterministic classification problem.

---

## 🗂️ Data
- **League:** English Premier League  
- **Seasons:** 2023–2024, 2024–2025, 2025–2026 (ongoing)

### 📥 Source
Data is **web-scraped from FBRef**

### 📊 What the dataset contains
Beyond final scores, the dataset includes team-level performance stats such as:
- Goals & Expected Goals (**GF/GA, xG/xGA**)
- Possession
- Shooting (shots, shots on target, finishing indicators)
- Passing (completion %, progressive passes, passes into final third/box)
- Defensive actions (tackles+interceptions, blocks, errors)

---

## 🧠 Methodology
- Goal scoring is modeled using the **Poisson distribution**
- Two separate expected goal parameters (λ) are estimated:
  - **Home team goals**
  - **Away team goals**

From these distributions, joint scoreline probabilities are computed and aggregated to obtain:
- Match outcome probabilities (1X2)
- Over / Under goal probabilities
---

## 📤 Outputs
For a given match, the system returns:
- λ_home, λ_away (expected goals)
- scoreline probability matrix (up to a chosen max-goals cutoff)
- aggregated probabilities:
  - Home / Draw / Away
  - Over/Under total goals

---

## ⚠️ Limitations
- Assumes **independence** between home and away goal-scoring processes
- Does not account for:
  - Tactical adjustments
  - Lineups or injuries
  - In-game dynamics
- **Draw outcomes**, especially low-scoring draws, remain inherently difficult to predict

---

## ✅ Conclusion
Poisson models are a strong and interpretable **statistical baseline** for football prediction.
This project focuses on producing **calibrated probability estimates** rather than “correct winner” guesses.
