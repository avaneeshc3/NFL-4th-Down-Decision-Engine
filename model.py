import nflreadpy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle


def preprocess_data():
    # Load all available play-by-play historical data (from 1999)
    pbp_df = nflreadpy.load_pbp(seasons=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]).to_pandas()
    pbp_df = pbp_df.drop_duplicates()
    pbp_df = pbp_df.dropna(subset=["down", "field_goal_attempt", "punt_attempt"])

    # Filter to respective play type attempts for each 4th down option - go for it, kick field goal, or punt
    conversion_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] != 1) & (pbp_df["punt_attempt"] != 1)]
    field_goal_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] == 1)]
    punt_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["punt_attempt"] == 1)]

    return conversion_df, field_goal_df, punt_df


def prepare_scenario_features(
    yards_to_go,
    yardline,
    team_run_tendency,
    team_pass_tendency,
    off_def_strength_diff,
    team_recent_epa,
    opp_recent_epa,
    is_goal_to_go,
    is_red_zone,
    short_yardage,
    quarter,
    seconds_remaining,
    score_diff,
    is_home,
    wp,
    def_wp,
    wp_delta,
    team_recent_success,
    opp_recent_success,
):
    return pd.DataFrame(
        [
            {
                "yards_to_go": yards_to_go,
                "yardline": yardline,
                "team_run_tendency": team_run_tendency,
                "team_pass_tendency": team_pass_tendency,
                "off_def_strength_diff": off_def_strength_diff,
                "team_recent_epa": team_recent_epa,
                "opp_recent_epa": opp_recent_epa,
                "is_goal_to_go": is_goal_to_go,
                "is_red_zone": is_red_zone,
                "short_yardage": short_yardage,
                "quarter": quarter,
                "seconds_remaining": seconds_remaining,
                "score_diff": score_diff,
                "is_home": is_home,
                "wp": wp,
                "def_wp": def_wp,
                "wp_delta": wp_delta,
                "team_recent_success": team_recent_success,
                "opp_recent_success": opp_recent_success,
            }
        ]
    )


def predict_success_probability(model, scenario_df, feature_names):
    if isinstance(scenario_df, dict):
        scenario_df = pd.DataFrame([scenario_df])
    feature_set = scenario_df[feature_names].copy().fillna(0)
    return float(model.predict_proba(feature_set)[:, 1][0])


def estimate_wp_after_outcome(current_state, action, outcome):
    return


def calculate_expected_wp(prob_success, wp_if_success, wp_if_failure):
    return float(prob_success * wp_if_success + (1 - prob_success) * wp_if_failure)


def build_conversion_model(conversion_df):
    model_df = conversion_df.copy()
    model_df = model_df.dropna(
        subset=[
            "season",
            "posteam",
            "defteam",
            "ydstogo",
            "yardline_100",
            "yards_gained",
            "epa",
            "qtr",
            "game_seconds_remaining",
            "score_differential",
            "goal_to_go",
            "wp",
        ]
    )

    # Feature engineering
    model_df["yards_to_go"] = model_df["ydstogo"].astype(float)
    model_df["yardline"] = model_df["yardline_100"].astype(float)

    model_df["is_rush"] = model_df["rush_attempt"].fillna(0).astype(float)
    model_df["is_pass"] = model_df["pass_attempt"].fillna(0).astype(float)
    model_df["team_rush_attempts"] = model_df.groupby(["season", "posteam"])["is_rush"].cumsum().shift(1).fillna(0)
    model_df["team_pass_attempts"] = model_df.groupby(["season", "posteam"])["is_pass"].cumsum().shift(1).fillna(0)
    model_df["team_total_plays"] = model_df["team_rush_attempts"] + model_df["team_pass_attempts"]
    model_df["team_run_tendency"] = (model_df["team_rush_attempts"] / model_df["team_total_plays"].replace(0, np.nan)).fillna(0.5)
    model_df["team_pass_tendency"] = 1 - model_df["team_run_tendency"]

    model_df["epa_value"] = model_df["epa"].fillna(0).astype(float)
    model_df["team_play_count"] = model_df.groupby(["season", "posteam"]).cumcount()
    model_df["opp_play_count"] = model_df.groupby(["season", "defteam"]).cumcount()
    model_df["team_off_epa"] = model_df.groupby(["season", "posteam"])["epa_value"].cumsum().shift(1).fillna(0) / model_df["team_play_count"].replace(0, 1)
    model_df["opp_epa_allowed"] = model_df.groupby(["season", "defteam"])["epa_value"].cumsum().shift(1).fillna(0) / model_df["opp_play_count"].replace(0, 1)
    model_df["off_def_strength_diff"] = model_df["team_off_epa"] - model_df["opp_epa_allowed"]
    model_df["team_recent_epa"] = model_df.groupby(["season", "posteam"])["epa_value"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    model_df["opp_recent_epa"] = model_df.groupby(["season", "defteam"])["epa_value"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    model_df["is_goal_to_go"] = (model_df["goal_to_go"] == 1).astype(int)
    model_df["is_red_zone"] = (model_df["yardline"] <= 20).astype(int)
    model_df["short_yardage"] = (model_df["yards_to_go"] <= 2).astype(int)
    model_df["quarter"] = model_df["qtr"].fillna(5).astype(float)
    model_df["seconds_remaining"] = model_df["game_seconds_remaining"].fillna(1800).astype(float)
    model_df["score_diff"] = model_df["score_differential"].fillna(0).astype(float)
    model_df["is_home"] = (model_df["posteam"] == model_df["home_team"]).astype(int)
    model_df["wp"] = model_df["wp"].fillna(0.5).astype(float)
    model_df["def_wp"] = model_df["def_wp"].fillna(0.5).astype(float)
    model_df["wp_delta"] = model_df["wp"] - model_df["def_wp"]

    model_df["success"] = ((model_df["yards_gained"].fillna(0) >= model_df["yards_to_go"]) | (model_df["touchdown"].fillna(0) == 1)).astype(int)
    model_df["team_recent_success"] = model_df.groupby(["season", "posteam"])["success"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    model_df["opp_recent_success"] = model_df.groupby(["season", "defteam"])["success"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    features = [
        "yards_to_go",
        "yardline",
        "team_run_tendency",
        "team_pass_tendency",
        "off_def_strength_diff",
        "team_recent_epa",
        "opp_recent_epa",
        "is_goal_to_go",
        "is_red_zone",
        "short_yardage",
        "quarter",
        "seconds_remaining",
        "score_diff",
        "is_home",
        "wp",
        "def_wp",
        "wp_delta",
        "team_recent_success",
        "opp_recent_success",
    ]
    feature_df = model_df[features].copy().fillna(0)
    y = model_df["success"]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, stratify=y, random_state=42)
    # Build RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Conversion model ROC-AUC score:", roc_auc_score(y_test, y_prob))

    return model, features


def build_field_goal_model(field_goal_df):
    return


def build_punt_model(punt_df):
    return


def main():
    conversion_df, field_goal_df, punt_df = preprocess_data()

    conversion_model, conversion_features = build_conversion_model(conversion_df)
    with open("conversion_model.pkl", "wb") as f:
        pickle.dump({"model": conversion_model, "features": conversion_features, "target_name": "conversion_success"}, f)


if __name__ == "__main__":
    main()
