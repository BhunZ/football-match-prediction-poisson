import argparse
from pathlib import Path
import pandas as pd


# Rename maps
RENAME_MAP_PASSING = {
    "Unnamed: 0_level_0_Squad": "team",
    "Unnamed: 1_level_0_# Pl": "players_used",
    "Unnamed: 2_level_0_90s": "minutes_90s",

    "Total_Cmp": "passes_completed",
    "Total_Att": "passes_attempted",
    "Total_Cmp%": "passes_completion_pct",
    "Total_TotDist": "passes_total_distance",
    "Total_PrgDist": "passes_progressive_distance",

    "Short_Cmp": "passes_completed_short",
    "Short_Att": "passes_attempted_short",
    "Short_Cmp%": "passes_completion_pct_short",

    "Medium_Cmp": "passes_completed_medium",
    "Medium_Att": "passes_attempted_medium",
    "Medium_Cmp%": "passes_completion_pct_medium",

    "Long_Cmp": "passes_completed_long",
    "Long_Att": "passes_attempted_long",
    "Long_Cmp%": "passes_completion_pct_long",

    "Unnamed: 17_level_0_Ast": "assists",
    "Unnamed: 18_level_0_xAG": "xag",
    "Expected_xA": "xa",
    "Expected_A-xAG": "assists_minus_xag",
    "Unnamed: 21_level_0_KP": "key_passes",
    "Unnamed: 22_level_0_1/3": "passes_into_final_third",
    "Unnamed: 23_level_0_PPA": "passes_into_penalty_area",
    "Unnamed: 24_level_0_CrsPA": "crosses_into_penalty_area",
    "Unnamed: 25_level_0_PrgP": "progressive_passes",

    "Season": "season",
    "StatType": "stat_type",
}

RENAME_MAP_POSSESSION = {
    "Unnamed: 0_level_0_Squad": "team",
    "Unnamed: 1_level_0_# Pl": "players_used",
    "Unnamed: 2_level_0_Poss": "possession",
    "Unnamed: 3_level_0_90s": "minutes_90s",

    "Touches_Touches": "touches",
    "Touches_Def Pen": "touches_def_pen_area",
    "Touches_Def 3rd": "touches_def_3rd",
    "Touches_Mid 3rd": "touches_mid_3rd",
    "Touches_Att 3rd": "touches_att_3rd",
    "Touches_Att Pen": "touches_att_pen_area",
    "Touches_Live": "touches_live_ball",

    "Take-Ons_Att": "take_ons",
    "Take-Ons_Succ": "take_ons_won",
    "Take-Ons_Succ%": "take_ons_won_pct",
    "Take-Ons_Tkld": "take_ons_tackled",
    "Take-Ons_Tkld%": "take_ons_tackled_pct",

    "Carries_Carries": "carries",
    "Carries_TotDist": "carries_distance",
    "Carries_PrgDist": "carries_progressive_distance",
    "Carries_PrgC": "progressive_carries",
    "Carries_1/3": "carries_into_final_third",
    "Carries_CPA": "carries_into_penalty_area",
    "Carries_Mis": "miscontrols",
    "Carries_Dis": "dispossessed",

    "Receiving_Rec": "passes_received",
    "Receiving_PrgrR": "progressive_passes_received",
    "Receiving_PrgR": "progressive_passes_received",

    "Season": "season",
    "StatType": "stat_type",
}

RENAME_MAP_DEFENSE = {
    "Unnamed: 0_level_0_Squad": "team",
    "Unnamed: 1_level_0_# Pl": "players_used",
    "Unnamed: 2_level_0_90s": "minutes_90s",

    "Tackles_Tkl": "tackles",
    "Tackles_TklW": "tackles_won",
    "Tackles_Def 3rd": "tackles_defensive_third",
    "Tackles_Mid 3rd": "tackles_middle_third",
    "Tackles_Att 3rd": "tackles_attacking_third",

    "Challenges_Tkl": "dribblers_tackled",
    "Challenges_Att": "dribble_challenges",
    "Challenges_Tkl%": "dribblers_tackled_pct",
    "Challenges_Lost": "dribble_challenges_lost",

    "Blocks_Blocks": "blocks",
    "Blocks_Sh": "blocked_shots",
    "Blocks_Pass": "blocked_passes",

    "Unnamed: 15_level_0_Int": "interceptions",
    "Unnamed: 16_level_0_Tkl+Int": "tackles_plus_interceptions",
    "Unnamed: 17_level_0_Clr": "clearances",
    "Unnamed: 18_level_0_Err": "errors_leading_to_shot",

    "Season": "season",
    "StatType": "stat_type",
}

RENAME_MAP_SHOOTING = {
    "Unnamed: 0_level_0_Squad": "team",
    "Unnamed: 1_level_0_# Pl": "players_used",
    "Unnamed: 2_level_0_90s": "minutes_90s",

    "Standard_Gls": "goals",
    "Standard_Sh": "shots",
    "Standard_SoT": "shots_on_target",
    "Standard_SoT%": "shots_on_target_pct",
    "Standard_Sh/90": "shots_per90",
    "Standard_SoT/90": "shots_on_target_per90",
    "Standard_G/Sh": "goals_per_shot",
    "Standard_G/SoT": "goals_per_shot_on_target",
    "Standard_Dist": "average_shot_distance",
    "Standard_FK": "shots_free_kicks",
    "Standard_PK": "pens_made",
    "Standard_PKatt": "pens_att",

    "Expected_xG": "xg",
    "Expected_npxG": "npxg",
    "Expected_npxG/Sh": "npxg_per_shot",
    "Expected_G-xG": "goals_minus_xg",
    "Expected_np:G-xG": "non_penalty_goals_minus_npxg",

    "Season": "season",
    "StatType": "stat_type",
}

RENAME_MAP_TEAM_STATS = {
    "Unnamed: 0_level_0_Squad": "team",
    "Unnamed: 1_level_0_# Pl": "players_used",
    "Unnamed: 2_level_0_Age": "avg_age",
    "Unnamed: 3_level_0_Poss": "possession_pct",

    "Playing Time_MP": "matches_played",
    "Playing Time_Starts": "starts",
    "Playing Time_Min": "minutes_played",
    "Playing Time_90s": "minutes_90s",

    "Performance_Gls": "goals",
    "Performance_Ast": "assists",
    "Performance_G+A": "goals_assists",
    "Performance_G-PK": "non_penalty_goals",
    "Performance_PK": "penalties_scored",
    "Performance_PKatt": "penalties_attempted",
    "Performance_CrdY": "yellow_cards",
    "Performance_CrdR": "red_cards",

    "Expected_xG": "xg",
    "Expected_npxG": "npxg",
    "Expected_xAG": "xg_assist",
    "Expected_npxG+xAG": "npxg_plus_xg_assist",

    "Progression_PrgC": "progressive_carries",
    "Progression_PrgP": "progressive_passes",

    "Per 90 Minutes_Gls": "goals_per90",
    "Per 90 Minutes_Ast": "assists_per90",
    "Per 90 Minutes_G+A": "goals_assists_per90",
    "Per 90 Minutes_G-PK": "non_penalty_goals_per90",
    "Per 90 Minutes_G+A-PK": "non_penalty_goals_assists_per90",

    "Per 90 Minutes_xG": "xg_per90",
    "Per 90 Minutes_xAG": "xg_assist_per90",
    "Per 90 Minutes_xG+xAG": "xg_plus_xg_assist_per90",
    "Per 90 Minutes_npxG": "npxg_per90",
    "Per 90 Minutes_npxG+xAG": "npxg_plus_xg_assist_per90",

    "Season": "season",
    "StatType": "stat_type",
}




def standardize_one(in_path: Path, out_path: Path, rename_map: dict) -> None:
    df = pd.read_csv(in_path)
    before_cols = list(df.columns)

    df = df.rename(columns=rename_map)

    # Report missing keys (optional but useful)
    missing_keys = [k for k in rename_map.keys() if k not in before_cols]
    if missing_keys:
        print(f"[WARN] {in_path.name}: {len(missing_keys)} mapping keys not found in file (maybe already standardized or slightly different).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] {in_path.name} -> {out_path} | shape={df.shape}")


def main():
    parser = argparse.ArgumentParser(description="Standardize FBRef table column names (full mapping).")
    parser.add_argument("--in-dir", type=str, default="data/raw", help="Input folder containing raw FBRef CSVs.")
    parser.add_argument("--out-dir", type=str, default="data/processed/standardized", help="Output folder for standardized CSVs.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    tasks = [
        ("pl_passing.csv", RENAME_MAP_PASSING),
        ("pl_possession.csv", RENAME_MAP_POSSESSION),
        ("pl_defense.csv", RENAME_MAP_DEFENSE),
        ("pl_shooting.csv", RENAME_MAP_SHOOTING),
        ("pl_team_stats.csv", RENAME_MAP_TEAM_STATS),
    ]

    for fname, rmap in tasks:
        in_path = in_dir / fname
        if not in_path.exists():
            print(f"[SKIP] Missing {in_path}")
            continue
        out_path = out_dir / fname
        standardize_one(in_path, out_path, rmap)


if __name__ == "__main__":
    main()
