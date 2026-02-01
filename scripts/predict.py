# scripts/predict.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson


# -----------------------------
# Helpers
# -----------------------------
def parse_score(score: str):
    if pd.isna(score):
        return (np.nan, np.nan)
    parts = re.split(r"[-–—]", str(score))
    if len(parts) < 2:
        return (np.nan, np.nan)
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return (np.nan, np.nan)


def bar(p, width=26):
    p = float(np.clip(p, 0, 1))
    filled = int(round(p * width))
    return "█" * filled + "░" * (width - filled)


def outcome_probs_from_lambdas(lam_home, lam_away, max_goals=10):
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)

    p_h = poisson.pmf(hg, lam_home)
    p_a = poisson.pmf(ag, lam_away)

    M = np.outer(p_h, p_a)

    p_draw = np.trace(M)
    p_home = np.tril(M, -1).sum()
    p_away = np.triu(M, 1).sum()

    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


def print_teams_table(teams, cols=5, width=18):
    print("Available teams in dataset:")
    print("-" * (cols * (width + 2)))
    for i in range(0, len(teams), cols):
        row = teams[i:i + cols]
        print("  ".join(f"{t:<{width}}" for t in row))
    print("-" * (cols * (width + 2)))
    print()


def load_inference_df(feat_path: Path, match_path: Path) -> pd.DataFrame:
    feat = pd.read_csv(feat_path)
    matches = pd.read_csv(match_path)

    feat["Date"] = pd.to_datetime(feat["Date"])
    matches["Date"] = pd.to_datetime(matches["Date"])

    hg, ag = zip(*matches["Score"].map(parse_score))
    matches["home_goals"] = hg
    matches["away_goals"] = ag

    df = feat.merge(
        matches[["Season", "Date", "Home", "Away", "home_goals", "away_goals"]],
        on=["Season", "Date", "Home", "Away"],
        how="inner",
    )
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    return df


def build_Xrow_for_any_fixture(df, X_cols, home, away, asof_date=None, median_source=None, n_recent=3):
    # cutoff date (avoid leakage)
    if asof_date is None:
        cutoff = df["Date"].max() + pd.Timedelta(days=1)
    else:
        cutoff = pd.to_datetime(asof_date)

    home_hist = df[(df["Home"] == home) & (df["Date"] < cutoff)].sort_values("Date").tail(n_recent)
    away_hist = df[(df["Away"] == away) & (df["Date"] < cutoff)].sort_values("Date").tail(n_recent)

    if len(home_hist) == 0:
        raise ValueError(f"Không có lịch sử sân nhà cho đội: {home} (trước {cutoff.date()})")
    if len(away_hist) == 0:
        raise ValueError(f"Không có lịch sử sân khách cho đội: {away} (trước {cutoff.date()})")

    # mean over last n_recent matches
    home_row = home_hist.select_dtypes(include=[np.number]).mean()
    away_row = away_hist.select_dtypes(include=[np.number]).mean()

    X_row = pd.Series(index=X_cols, dtype=float)

    # fill with medians (fallback) so model/pipeline can handle missing
    if median_source is None:
        med = pd.Series(0.0, index=X_cols)
    else:
        med = median_source.median(numeric_only=True).reindex(X_cols)
        med = med.fillna(0.0)

    X_row[:] = med.values

    # fill home_*, away_*
    for c in X_cols:
        if c.startswith("home_"):
            X_row[c] = home_row.get(c, X_row[c])
        elif c.startswith("away_"):
            X_row[c] = away_row.get(c, X_row[c])

    # recompute diff_*
    for c in X_cols:
        if c.startswith("diff_"):
            base = c[len("diff_"):]
            hc = "home_" + base
            ac = "away_" + base
            if hc in X_cols and ac in X_cols:
                if pd.notna(X_row[hc]) and pd.notna(X_row[ac]):
                    X_row[c] = X_row[hc] - X_row[ac]

    # recompute rest_diff
    if "rest_diff" in X_cols and "home_rest_days" in X_cols and "away_rest_days" in X_cols:
        if pd.notna(X_row["home_rest_days"]) and pd.notna(X_row["away_rest_days"]):
            X_row["rest_diff"] = X_row["home_rest_days"] - X_row["away_rest_days"]

    return pd.DataFrame([X_row], columns=X_cols)


def pretty_print_result(home, away, asof_date, lam_home, lam_away, pH, pD, pA, p_over, p_under, pred_label):
    print("┌" + "─" * 62 + "┐")
    print(f"│  Model:   {'Poisson Goals (goals → 1X2 + O/U 2.5)':<53}│")
    print(f"│  Fixture: {home} vs {away:<44}│")
    if asof_date is None:
        print(f"│  As-of:   latest available in dataset{' ' * 28}│")
    else:
        d = pd.to_datetime(asof_date).date()
        pad = 62 - len("│  As-of:   ") - len(str(d)) - 1
        print(f"│  As-of:   {d}{' ' * pad}│")
    print("├" + "─" * 62 + "┤")

    lam_total = lam_home + lam_away
    print(f"│  λ_home: {lam_home:6.3f}   λ_away: {lam_away:6.3f}   λ_total: {lam_total:6.3f} │")
    print("├" + "─" * 62 + "┤")
    print(f"│  HOME WIN  {pH*100:6.2f}%  {bar(pH):<26}            │")
    print(f"│  DRAW      {pD*100:6.2f}%  {bar(pD):<26}            │")
    print(f"│  AWAY WIN  {pA*100:6.2f}%  {bar(pA):<26}            │")
    print("├" + "─" * 62 + "┤")
    print(f"│  OVER  2.5 {p_over*100:6.2f}%  {bar(p_over):<26}            │")
    print(f"│  UNDER 2.5 {p_under*100:6.2f}%  {bar(p_under):<26}            │")
    print("├" + "─" * 62 + "┤")
    print(f"│  Prediction: {pred_label:<48}│")
    print("└" + "─" * 62 + "┘")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict match outcome (Poisson) with team list display.")
    parser.add_argument("--feat-path", type=str, default="data/processed/pl_training_dataset.csv")
    parser.add_argument("--match-path", type=str, default="data/raw/pl_matches.csv")
    parser.add_argument("--model-dir", type=str, default="models/poisson")

    # CLI mode (optional). If not provided -> interactive prompts.
    parser.add_argument("--home", type=str, default=None)
    parser.add_argument("--away", type=str, default=None)
    parser.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD; omit for latest")
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--n-recent", type=int, default=3, help="Use last N matches (home at home, away away) to build features.")
    args = parser.parse_args()

    feat_path = Path(args.feat_path)
    match_path = Path(args.match_path)
    model_dir = Path(args.model_dir)

    home_model_path = model_dir / "poisson_home_model.joblib"
    away_model_path = model_dir / "poisson_away_model.joblib"
    cols_path = model_dir / "poisson_feature_columns.joblib"

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")
    if not match_path.exists():
        raise FileNotFoundError(f"Missing: {match_path}")
    if not home_model_path.exists() or not away_model_path.exists() or not cols_path.exists():
        raise FileNotFoundError("Missing model files. Run training first (scripts/train_poisson.py).")

    print("Loading dataset for inference ...")
    df = load_inference_df(feat_path, match_path)

    teams = sorted(set(df["Home"]).union(set(df["Away"])))
    print(f"✅ Loaded df: {df.shape[0]} matches | {len(teams)} teams\n")
    print_teams_table(teams, cols=5, width=18)

    # Choose home/away
    home = args.home
    away = args.away
    asof_date = None if (args.asof is None or str(args.asof).strip() == "") else args.asof

    if home is None or away is None:
        home = input("Home team (copy from list above): ").strip()
        away = input("Away team (copy from list above): ").strip()
        asof_in = input("As-of date YYYY-MM-DD [enter for latest]: ").strip()
        asof_date = None if asof_in == "" else asof_in

    if home not in teams:
        print(f"⚠️ Home team '{home}' not found in dataset teams.")
    if away not in teams:
        print(f"⚠️ Away team '{away}' not found in dataset teams.")
    print()

    # Load models + columns
    home_model = joblib.load(home_model_path)
    away_model = joblib.load(away_model_path)
    cols = joblib.load(cols_path)

    # Build median source for fallback fill
    X_full = df.reindex(columns=cols).select_dtypes(include=[np.number])

    X_row = build_Xrow_for_any_fixture(
        df=df,
        X_cols=cols,
        home=home,
        away=away,
        asof_date=asof_date,
        median_source=X_full,
        n_recent=args.n_recent,
    )

    lam_home = float(home_model.predict(X_row)[0])
    lam_away = float(away_model.predict(X_row)[0])
    lam_total = lam_home + lam_away

    p_under_25 = float(poisson.cdf(2, lam_total))
    p_over_25 = float(1.0 - p_under_25)

    pH, pD, pA = outcome_probs_from_lambdas(lam_home, lam_away, max_goals=args.max_goals)
    probs = {f"{home} WIN": pH, "DRAW": pD, f"{away} WIN": pA}
    pred_label = max(probs, key=probs.get)

    pretty_print_result(home, away, asof_date, lam_home, lam_away, pH, pD, pA, p_over_25, p_under_25, pred_label)


if __name__ == "__main__":
    main()
