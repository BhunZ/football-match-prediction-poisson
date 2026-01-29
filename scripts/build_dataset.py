# scripts/build_dataset.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROLL_WINDOWS = (5,)   # last 5 matches
EWM_ALPHA = 0.30


# -----------------------------
# Utilities
# -----------------------------
def parse_score(score: str):
    """Parse score string like '2–1' or '2-1' into (home_goals, away_goals)."""
    if pd.isna(score):
        return (np.nan, np.nan)
    s = str(score).strip()
    parts = re.split(r"[–-]", s)  # en-dash or hyphen
    if len(parts) != 2:
        return (np.nan, np.nan)
    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except Exception:
        return (np.nan, np.nan)


def prev_season(s: str) -> str:
    """Map season '2024-2025' -> '2023-2024'."""
    y1, y2 = s.split("-")
    return f"{int(y1)-1}-{int(y2)-1}"


def add_rolling_features(df_team: pd.DataFrame, windows=ROLL_WINDOWS, alpha=EWM_ALPHA) -> pd.DataFrame:
    df_team = df_team.sort_values("Date").copy()

    for w in windows:
        for col in ["GF", "GA", "GD", "xGF", "xGA", "xGD", "Pts"]:
            shifted = df_team[col].shift(1)
            df_team[f"roll{w}_{col.lower()}_mean"] = shifted.rolling(w, min_periods=1).mean()
            df_team[f"roll{w}_{col.lower()}_sum"] = shifted.rolling(w, min_periods=1).sum()

    for col in ["xGF", "xGA", "Pts", "GF", "GA"]:
        df_team[f"ewm_{col.lower()}"] = df_team[col].shift(1).ewm(alpha=alpha, adjust=False).mean()

    # Rest days: fill first match with team median rest
    df_team["rest_days"] = df_team["Date"].diff().dt.days
    df_team["rest_days"] = df_team["rest_days"].fillna(df_team["rest_days"].median())

    # Lag result one-hot
    df_team["win_last"] = (df_team["Result"].shift(1) == "W").astype(int)
    df_team["draw_last"] = (df_team["Result"].shift(1) == "D").astype(int)
    df_team["loss_last"] = (df_team["Result"].shift(1) == "L").astype(int)

    return df_team


def _ensure_season_col(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure Season exists (standardized tables usually use 'season')."""
    if "season" in df.columns and "Season" not in df.columns:
        df = df.rename(columns={"season": "Season"})
    return df


def _to_priors_key(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize team key to 'Squad' for joining priors."""
    df = df.copy()
    if "team" in df.columns and "Squad" not in df.columns:
        df = df.rename(columns={"team": "Squad"})
    return df


# -----------------------------
# Main pipeline
# -----------------------------
def build_dataset(raw_dir: Path, std_dir: Path, out_path: Path) -> pd.DataFrame:
    # Required files
    req_raw = ["pl_matches.csv"]
    req_std = ["pl_team_stats.csv", "pl_shooting.csv", "pl_defense.csv", "pl_passing.csv"]

    missing_raw = [f for f in req_raw if not (raw_dir / f).exists()]
    missing_std = [f for f in req_std if not (std_dir / f).exists()]

    if missing_raw or missing_std:
        msg = []
        if missing_raw:
            msg.append("Missing required RAW files:")
            msg += [f"- {m}" for m in missing_raw]
            msg.append(f"Expected in: {raw_dir.resolve()}")
        if missing_std:
            msg.append("\nMissing required STANDARDIZED files:")
            msg += [f"- {m}" for m in missing_std]
            msg.append(f"Expected in: {std_dir.resolve()}")
        raise FileNotFoundError("\n".join(msg))

    # ---- Load matches from RAW ----
    matches = pd.read_csv(raw_dir / "pl_matches.csv")
    matches["Date"] = pd.to_datetime(matches["Date"])

    # Ensure Season exists
    matches = _ensure_season_col(matches)

    # Score -> goals
    matches[["home_goals", "away_goals"]] = matches["Score"].apply(lambda x: pd.Series(parse_score(x)))

    # xG columns from FBRef export format
    if "xG" not in matches.columns or "xG.1" not in matches.columns:
        raise ValueError("pl_matches.csv must contain columns: 'xG' and 'xG.1' (home/away xG).")

    matches["home_xG"] = matches["xG"]
    matches["away_xG"] = matches["xG.1"]

    # label: 0 HomeWin, 1 Draw, 2 AwayWin
    matches["y"] = np.where(
        matches["home_goals"] > matches["away_goals"],
        0,
        np.where(matches["home_goals"] == matches["away_goals"], 1, 2),
    )

    # Sort chronologically (safe)
    sort_cols = [c for c in ["Season", "Date", "Time", "Wk"] if c in matches.columns]
    matches = matches.sort_values(sort_cols).reset_index(drop=True)

    # ---- Build team-long table ----
    home_rows = pd.DataFrame(
        {
            "Season": matches["Season"],
            "Date": matches["Date"],
            "Wk": matches["Wk"],
            "Team": matches["Home"],
            "Opponent": matches["Away"],
            "is_home": 1,
            "GF": matches["home_goals"],
            "GA": matches["away_goals"],
            "xGF": matches["home_xG"],
            "xGA": matches["away_xG"],
        }
    )

    away_rows = pd.DataFrame(
        {
            "Season": matches["Season"],
            "Date": matches["Date"],
            "Wk": matches["Wk"],
            "Team": matches["Away"],
            "Opponent": matches["Home"],
            "is_home": 0,
            "GF": matches["away_goals"],
            "GA": matches["home_goals"],
            "xGF": matches["away_xG"],
            "xGA": matches["home_xG"],
        }
    )

    team_long = pd.concat([home_rows, away_rows], ignore_index=True)
    team_long["GD"] = team_long["GF"] - team_long["GA"]
    team_long["xGD"] = team_long["xGF"] - team_long["xGA"]

    team_long["Pts"] = np.select(
        [team_long["GF"] > team_long["GA"], team_long["GF"] == team_long["GA"]],
        [3, 1],
        default=0,
    )

    team_long["Result"] = np.select(
        [team_long["GF"] > team_long["GA"], team_long["GF"] == team_long["GA"]],
        ["W", "D"],
        default="L",
    )

    team_long = team_long.sort_values(["Season", "Team", "Date"]).reset_index(drop=True)

    # ---- Rolling / EWM / rest / lag ----
    gb = team_long.groupby(["Season", "Team"], group_keys=False)

    # pandas mới có thể drop grouping columns trong apply -> cần ép giữ lại
    try:
        team_feat = gb.apply(add_rolling_features, include_groups=True)
    except TypeError:
        team_feat = gb.apply(add_rolling_features)
    
    # nếu Season/Team vẫn không nằm trong columns thì kéo từ index ra
    if ("Season" not in team_feat.columns) or ("Team" not in team_feat.columns):
        team_feat = team_feat.reset_index()
    
    team_feat = team_feat.reset_index(drop=True)


    feat_cols = [
        c for c in team_feat.columns
        if c.startswith(("roll", "ewm", "rest_days", "win_last", "draw_last", "loss_last"))
    ]

    home_feat = team_feat[team_feat["is_home"] == 1][["Season", "Date", "Team"] + feat_cols].copy()
    away_feat = team_feat[team_feat["is_home"] == 0][["Season", "Date", "Team"] + feat_cols].copy()

    home_feat = home_feat.rename(columns={"Team": "Home", **{c: f"home_{c}" for c in feat_cols}})
    away_feat = away_feat.rename(columns={"Team": "Away", **{c: f"away_{c}" for c in feat_cols}})

    dataset = matches[["Season", "Date", "Wk", "Home", "Away", "y"]].merge(
        home_feat, on=["Season", "Date", "Home"], how="left"
    )
    dataset = dataset.merge(away_feat, on=["Season", "Date", "Away"], how="left")

    # ---- Load priors from STANDARDIZED tables ----
    team_stats = pd.read_csv(std_dir / "pl_team_stats.csv")
    shooting   = pd.read_csv(std_dir / "pl_shooting.csv")
    defense    = pd.read_csv(std_dir / "pl_defense.csv")
    passing    = pd.read_csv(std_dir / "pl_passing.csv")

    for d in [team_stats, shooting, defense, passing]:
        d[:] = _ensure_season_col(d)

    team_stats = _to_priors_key(team_stats)
    shooting   = _to_priors_key(shooting)
    defense    = _to_priors_key(defense)
    passing    = _to_priors_key(passing)

    pri_team = team_stats[[
        "Season", "Squad",
        "goals_per90",
        "xg_per90",
        "npxg_per90",
        "xg_assist_per90",
        "possession_pct",
    ]].rename(columns={
        "goals_per90": "prior_goals90",
        "xg_per90": "prior_xg90",
        "npxg_per90": "prior_npxg90",
        "xg_assist_per90": "prior_xag90",
        "possession_pct": "prior_poss",
    })

    pri_shot = shooting[[
        "Season", "Squad",
        "shots_per90",
        "shots_on_target_per90",
        "npxg_per_shot",
        "goals_minus_xg",
    ]].rename(columns={
        "shots_per90": "prior_sh90",
        "shots_on_target_per90": "prior_sot90",
        "npxg_per_shot": "prior_npxg_per_sh",
        "goals_minus_xg": "prior_finishing_delta",
    })

    pri_def = defense[[
        "Season", "Squad",
        "tackles_plus_interceptions",
        "blocks",
        "errors_leading_to_shot",
    ]].rename(columns={
        "tackles_plus_interceptions": "prior_def_actions",
        "blocks": "prior_blocks",
        "errors_leading_to_shot": "prior_def_errors",
    })

    pri_pass = passing[[
        "Season", "Squad",
        "passes_completion_pct",
        "progressive_passes",
        "passes_into_final_third",
        "passes_into_penalty_area",
    ]].rename(columns={
        "passes_completion_pct": "prior_pass_pct",
        "progressive_passes": "prior_prog_pass",
        "passes_into_final_third": "prior_pass_final3rd",
        "passes_into_penalty_area": "prior_pass_box",
    })

    priors = (
        pri_team
        .merge(pri_shot, on=["Season", "Squad"], how="outer")
        .merge(pri_def,  on=["Season", "Squad"], how="outer")
        .merge(pri_pass, on=["Season", "Squad"], how="outer")
    )

    # ---- Join priors using previous season ----
    dataset["prev_season"] = dataset["Season"].apply(prev_season)

    home_pr = priors.rename(columns={"Season": "prev_season", "Squad": "Home"})
    away_pr = priors.rename(columns={"Season": "prev_season", "Squad": "Away"}).add_prefix("away_prior_")

    dataset = dataset.merge(home_pr, on=["prev_season", "Home"], how="left")
    dataset = dataset.merge(
        away_pr,
        left_on=["prev_season", "Away"],
        right_on=["away_prior_prev_season", "away_prior_Away"],
        how="left",
    )

    dataset = dataset.drop(
        columns=[c for c in dataset.columns if c.endswith("_prev_season") or c.endswith("_Away")],
        errors="ignore",
    )

    # ---- Relative features (diff) ----
    for c in feat_cols:
        dataset[f"diff_{c}"] = dataset[f"home_{c}"] - dataset[f"away_{c}"]
    dataset["rest_diff"] = dataset["home_rest_days"] - dataset["away_rest_days"]

    # ---- Fill numeric NaNs ----
    num_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[num_cols] = dataset[num_cols].fillna(dataset[num_cols].mean())

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build PL training dataset (rolling/ewm/priors/diff).")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw files (must include pl_matches.csv)."
    )
    parser.add_argument(
        "--std-dir",
        type=str,
        default="data/processed/standardized",
        help="Directory containing standardized FBRef tables (team/shooting/defense/passing)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/pl_training_dataset.csv",
        help="Output CSV path."
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    std_dir = Path(args.std_dir)
    out_path = Path(args.out)

    df = build_dataset(raw_dir, std_dir, out_path)
    print(f"[OK] Saved dataset: {out_path} | shape={df.shape}")


if __name__ == "__main__":
    main()
