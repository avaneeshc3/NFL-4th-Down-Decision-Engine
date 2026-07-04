import nflreadpy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle


def preprocess_data():
    # Load all available play-by-play historical data (from 1999)
    pbp_df = nflreadpy.load_pbp(seasons=True).to_pandas()
    pbp_df = pbp_df.drop_duplicates()
    pbp_df = pbp_df.dropna(subset=["down", "field_goal_attempt", "punt_attempt", "wp"])
    # Create a win probability after play column (using win probability at start of the next play) for expected win probability calculation
    pbp_df = pbp_df.sort_values(["game_id", "play_id"])
    pbp_df["wp_after"] = pbp_df.groupby("game_id")["wp"].shift(-1)

    # Filter to respective play type attempts for each play option - go for it, kick field goal, or punt
    conversion_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] != 1) & (pbp_df["punt_attempt"] != 1)]
    field_goal_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] == 1)]
    punt_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["punt_attempt"] == 1)]

    return conversion_df, field_goal_df, punt_df


def build_conversion_model(conversion_df):
    """
    Creates a model that predicts the probability of successfully converting to a 1st down in the situation.
    """
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
    model_df["wp_after"] = model_df["wp_after"].fillna(model_df["wp"])

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

    feature_df = model_df[features].fillna(0)
    y = model_df["success"]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, stratify=y, random_state=42)
    # Build Random Forest Classifier model
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
    """
    Creates a model that predicts the probability of making a field goal in the situation.
    """
    model_df = field_goal_df.copy()
    model_df = model_df.dropna(subset=["kick_distance", "field_goal_result"])

    # Feature engineering
    model_df["kick_distance"] = model_df["kick_distance"].astype(float)
    model_df["quarter"] = model_df["qtr"].fillna(5).astype(float)
    model_df["seconds_remaining"] = model_df["game_seconds_remaining"].fillna(1800).astype(float)
    model_df["score_diff"] = model_df["score_differential"].fillna(0).astype(float)
    model_df["is_home"] = (model_df["posteam"] == model_df["home_team"]).astype(int)
    model_df["wp"] = model_df["wp"].fillna(0.5).astype(float)
    model_df["temp"] = model_df["temp"].fillna(70).astype(float)
    model_df["wind"] = model_df["wind"].fillna(0).astype(float)

    model_df["success"] = (model_df["field_goal_result"] == "made").astype(int)

    features = [
        "kick_distance",
        "quarter",
        "seconds_remaining",
        "score_diff",
        "is_home",
        "wp",
        "temp",
        "wind",
    ]

    feature_df = model_df[features].fillna(0)
    y = model_df["success"]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, stratify=y, random_state=42)
    # Build Logistic Regression model (very well-suited for field goal success prediction)
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Field goal model ROC-AUC score:", roc_auc_score(y_test, y_prob))

    return model, features


def build_punt_model(punt_df):
    """
    Creates a model that predicts the probability of making a successful punt (opponent starts within their 25-yard line) in the situation.
    """
    model_df = punt_df.copy()
    model_df = model_df.dropna(
        subset=[
            "yardline_100",
            "kick_distance",
            "return_yards",
            "punt_blocked",
            "qtr",
            "game_seconds_remaining",
            "score_differential",
            "posteam",
            "home_team",
            "wp",
            "temp",
            "wind",
        ]
    )
    # Filter out blocked punts
    model_df = model_df[model_df["punt_blocked"].fillna(0) == 0]

    # Feature engineering
    model_df["yardline"] = model_df["yardline_100"].astype(float)
    model_df["kick_distance"] = model_df["kick_distance"].astype(float)
    model_df["quarter"] = model_df["qtr"].fillna(5).astype(float)
    model_df["seconds_remaining"] = model_df["game_seconds_remaining"].fillna(1800).astype(float)
    model_df["score_diff"] = model_df["score_differential"].fillna(0).astype(float)
    model_df["is_home"] = (model_df["posteam"] == model_df["home_team"]).astype(int)
    model_df["wp"] = model_df["wp"].fillna(0.5).astype(float)
    model_df["temp"] = model_df["temp"].fillna(70).astype(float)
    model_df["wind"] = model_df["wind"].fillna(0).astype(float)

    model_df["success"] = (model_df["yardline"] - model_df["kick_distance"] + model_df["return_yards"] <= 25).astype(int)

    features = [
        "yardline",
        "kick_distance",
        "quarter",
        "seconds_remaining",
        "score_diff",
        "is_home",
        "wp",
        "temp",
        "wind",
    ]

    feature_df = model_df[features].fillna(0)
    y = model_df["success"]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, stratify=y, random_state=42)
    # Build Random Forest Classifier model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Punt model ROC-AUC score:", roc_auc_score(y_test, y_prob))

    return model, features


def prepare_features(
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
    kick_distance,
    temp,
    wind
):
    # Format user-entered game state details as a DataFrame for model prediction
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
                "kick_distance": kick_distance,
                "temp": temp,
                "wind": wind
            }
        ]
    )


def predict_success_probability(model, state_df, feature_names):
    # Predict the success probability of the chosen play option using the relevant model and features
    features = state_df[feature_names].copy().fillna(0)
    return model.predict_proba(features)[:, 1][0]


def project_conversion_wp(state_df, outcome):
    state = state_df.iloc[0].to_dict()
    current_wp = state.get("wp", 0.5)
    yardline = state.get("yardline", 50)
    score_diff = state.get("score_diff", 0)
    is_goal_to_go = state.get("is_goal_to_go", 0)
    seconds_remaining = state.get("seconds_remaining", 1800)

    # Adjust win probability based on game state
    if outcome == "success":
        adjustment = 0.02
        if yardline <= 35:
            adjustment = 0.03
        if yardline <= 20:
            adjustment = 0.04
        if is_goal_to_go == 1:
            adjustment = 0.05
    else:
        adjustment = -0.02
        if yardline >= 40:
            adjustment = -0.03
        if yardline >= 55:
            adjustment = -0.04
        if score_diff < 0:
            adjustment -= 0.01
            if seconds_remaining <= 300:
                adjustment -= 0.01

    return max(min(current_wp + adjustment, 1), 0)


def project_field_goal_wp(state_df, outcome):
    state = state_df.iloc[0].to_dict()
    current_wp = state.get("wp", 0.5)
    yardline = state.get("yardline", 50)
    score_diff = state.get("score_diff", 0)
    quarter = state.get("quarter", 3)
    seconds_remaining = state.get("seconds_remaining", 1800)

    if outcome == "success":
        adjustment = 0.03
        if score_diff >= 4:
            adjustment = 0.05
        if quarter == 4:
            adjustment += 0.01
        if seconds_remaining <= 300:
            adjustment += 0.02

    else:
        adjustment = -0.03
        if yardline >= 15:
            adjustment = -0.04
        if yardline >= 30:
            adjustment = -0.05
        if quarter == 4:
            adjustment -= 0.01
        if score_diff < 0:
            adjustment -= 0.01
            if seconds_remaining <= 300:
                adjustment -= 0.01

    return max(min(current_wp + adjustment, 1), 0)


def project_punt_wp(state_df, outcome):
    state = state_df.iloc[0].to_dict()
    current_wp = state.get("wp", 0.5)
    score_diff = state.get("score_diff", 0)
    seconds_remaining = state.get("seconds_remaining", 1800)

    if outcome == "success":
        adjustment = 0.01
        if score_diff >= 0:
            adjustment = 0.02
        else:
            if seconds_remaining <= 300:
                adjustment = -0.02
    else:
        adjustment = -0.02
        if score_diff < 0:
            adjustment = -0.03
            if seconds_remaining <= 300:
                adjustment = -0.04

    return max(min(current_wp + adjustment, 1), 0)


def estimate_wp_after_outcome(state_df, action, outcome):
    if action == "conversion":
        return project_conversion_wp(state_df, outcome)
    elif action == "field_goal":
        return project_field_goal_wp(state_df, outcome)
    else:
        return project_punt_wp(state_df, outcome)


def calculate_expected_wp(prob_success, wp_if_success, wp_if_failure):
    return (prob_success * wp_if_success + (1 - prob_success) * wp_if_failure)


def main():
    """
    Receives the model and features for each play option, pickling them for predictions on user-entered game states.
    """
    conversion_df, field_goal_df, punt_df = preprocess_data()

    conversion_model, conversion_features = build_conversion_model(conversion_df)
    with open("conversion_model.pkl", "wb") as f:
        pickle.dump({"model": conversion_model, "features": conversion_features, "target_name": "conversion_probability"}, f)

    field_goal_model, field_goal_features = build_field_goal_model(field_goal_df)
    with open("field_goal_model.pkl", "wb") as f:
        pickle.dump({"model": field_goal_model, "features": field_goal_features, "target_name": "field_goal_probability"}, f)

    punt_model, punt_features = build_punt_model(punt_df)
    with open("punt_model.pkl", "wb") as f:
        pickle.dump({"model": punt_model, "features": punt_features, "target_name": "punt_probability"}, f)


if __name__ == "__main__":
    main()
