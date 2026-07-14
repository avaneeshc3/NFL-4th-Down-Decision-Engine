import nflreadpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random


def preprocess_data():
    # Load all available play-by-play historical data (from 1999)
    pbp_df = nflreadpy.load_pbp(seasons=True).to_pandas()
    pbp_df = pbp_df.drop_duplicates()
    pbp_df = pbp_df.dropna(subset=["down", "field_goal_attempt", "punt_attempt", "wp"])
    pbp_df = pbp_df.sort_values(["game_id", "play_id"])
    # Create a win probability after play column (using win probability at start of the next play) for expected win probability calculation
    pbp_df["wp_after"] = pbp_df.groupby("game_id")["wp"].shift(-1)

    # Filter to respective play type attempts for each 4th down play option - go for it, kick field goal, or punt
    conversion_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] != 1) & (pbp_df["punt_attempt"] != 1)]
    field_goal_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] == 1)]
    punt_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["punt_attempt"] == 1)]

    return conversion_df, field_goal_df, punt_df


def build_conversion_success_model(conversion_df):
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
            "wp"
        ]
    )

    # Feature engineering
    model_df["yards_to_go"] = model_df["ydstogo"]
    model_df["yardline"] = model_df["yardline_100"]

    model_df["is_rush"] = model_df["rush_attempt"].fillna(0)
    model_df["is_pass"] = model_df["pass_attempt"].fillna(0)
    model_df["team_rush_attempts"] = model_df.groupby(["season", "posteam"])["is_rush"].cumsum().shift(1).fillna(0)
    model_df["team_pass_attempts"] = model_df.groupby(["season", "posteam"])["is_pass"].cumsum().shift(1).fillna(0)
    model_df["team_total_plays"] = model_df["team_rush_attempts"] + model_df["team_pass_attempts"]
    model_df["team_run_tendency"] = (model_df["team_rush_attempts"] / model_df["team_total_plays"].replace(0, np.nan)).fillna(0.5)
    model_df["team_pass_tendency"] = 1 - model_df["team_run_tendency"]

    model_df["epa_value"] = model_df["epa"].fillna(0)
    model_df["team_play_count"] = model_df.groupby(["season", "posteam"]).cumcount()
    model_df["def_play_count"] = model_df.groupby(["season", "defteam"]).cumcount()
    model_df["team_epa"] = model_df.groupby(["season", "posteam"])["epa_value"].cumsum().shift(1).fillna(0) / model_df["team_play_count"].replace(0, 1)
    model_df["def_epa"] = model_df.groupby(["season", "defteam"])["epa_value"].cumsum().shift(1).fillna(0) / model_df["def_play_count"].replace(0, 1)
    model_df["epa_diff"] = model_df["team_epa"] - model_df["def_epa"]
    model_df["team_recent_epa"] = model_df.groupby(["season", "posteam"])["epa_value"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    model_df["def_recent_epa"] = model_df.groupby(["season", "defteam"])["epa_value"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    model_df["is_goal_to_go"] = (model_df["goal_to_go"] == 1).astype(int)
    model_df["is_red_zone"] = (model_df["yardline"] <= 20).astype(int)
    model_df["is_short_yardage"] = (model_df["yards_to_go"] <= 2).astype(int)
    model_df["quarter"] = model_df["qtr"].fillna(5)
    model_df["seconds_remaining"] = model_df["game_seconds_remaining"].fillna(1800)
    model_df["score_diff"] = model_df["score_differential"].fillna(0)
    model_df["is_home"] = (model_df["posteam"] == model_df["home_team"]).astype(int)
    model_df["wp"] = model_df["wp"].fillna(0.5)
    model_df["def_wp"] = model_df["def_wp"].fillna(0.5)
    model_df["wp_diff"] = model_df["wp"] - model_df["def_wp"]

    model_df["success"] = ((model_df["yards_gained"].fillna(0) >= model_df["yards_to_go"]) | (model_df["touchdown"].fillna(0) == 1)).astype(int)
    model_df["team_recent_conversion_success"] = model_df.groupby(["season", "posteam"])["success"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    model_df["def_recent_conversion_success"] = model_df.groupby(["season", "defteam"])["success"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    features = [
        "yards_to_go",
        "yardline",
        "team_run_tendency",
        "team_pass_tendency",
        "epa_diff",
        "team_recent_epa",
        "def_recent_epa",
        "is_goal_to_go",
        "is_red_zone",
        "is_short_yardage",
        "quarter",
        "seconds_remaining",
        "score_diff",
        "is_home",
        "wp",
        "def_wp",
        "wp_diff",
        "team_recent_conversion_success",
        "def_recent_conversion_success"
    ]

    feature_df = model_df[features].fillna(0)
    y = model_df["success"]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, stratify=y, random_state=42)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Conversion success model ROC-AUC score:", roc_auc_score(y_test, y_prob))

    return model, features, model_df


def build_field_goal_success_model(field_goal_df):
    """
    Creates a model that predicts the probability of making a field goal in the situation.
    """
    model_df = field_goal_df.copy()
    model_df = model_df.dropna(subset=["kick_distance", "field_goal_result"])

    # Feature engineering
    model_df["quarter"] = model_df["qtr"].fillna(5)
    model_df["seconds_remaining"] = model_df["game_seconds_remaining"].fillna(1800)
    model_df["score_diff"] = model_df["score_differential"].fillna(0)
    model_df["is_home"] = (model_df["posteam"] == model_df["home_team"]).astype(int)
    model_df["wp"] = model_df["wp"].fillna(0.5)
    model_df["temp"] = model_df["temp"].fillna(70)
    model_df["wind"] = model_df["wind"].fillna(0)

    model_df["success"] = (model_df["field_goal_result"] == "made").astype(int)

    features = [
        "kick_distance",
        "quarter",
        "seconds_remaining",
        "score_diff",
        "is_home",
        "wp",
        "temp",
        "wind"
    ]

    feature_df = model_df[features].fillna(0)
    y = model_df["success"]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Field goal success model ROC-AUC score:", roc_auc_score(y_test, y_prob))

    return model, features, model_df


def build_punt_success_model(punt_df):
    """
    Creates a model that predicts the probability of making a "successful" punt (opponent starts within their 25-yard line) in the situation.
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
            "wind"
        ]
    )
    # Filter out blocked punts
    model_df = model_df[model_df["punt_blocked"].fillna(0) == 0]

    # Feature engineering
    model_df["yardline"] = model_df["yardline_100"]
    model_df["kick_distance"] = model_df["kick_distance"]
    model_df["quarter"] = model_df["qtr"].fillna(5)
    model_df["seconds_remaining"] = model_df["game_seconds_remaining"].fillna(1800)
    model_df["score_diff"] = model_df["score_differential"].fillna(0)
    model_df["is_home"] = (model_df["posteam"] == model_df["home_team"]).astype(int)
    model_df["wp"] = model_df["wp"].fillna(0.5)
    model_df["temp"] = model_df["temp"].fillna(70)
    model_df["wind"] = model_df["wind"].fillna(0)

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
        "wind"
    ]

    feature_df = model_df[features].fillna(0)
    y = model_df["success"]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, stratify=y, random_state=42)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Punt success model ROC-AUC score:", roc_auc_score(y_test, y_prob))

    return model, features, model_df


def predict_success_prob(model, current_state, features):
    feature_df = current_state[features].copy()
    return model.predict_proba(feature_df)[:, 1][0]


def create_wp_features(df):
    df["yardline"] = df["yardline_100"]
    df["quarter"] = df["qtr"].fillna(5)
    df["seconds_remaining"] = df["game_seconds_remaining"].fillna(1800)
    df["score_diff"] = df["score_differential"].fillna(0)
    df["is_home"] = (df["posteam"] == df["home_team"]).astype(int)
    df["is_goal_to_go"] = (df["goal_to_go"] == 1).astype(int)
    df["is_red_zone"] = (df["yardline"] <= 20).astype(int)
    df["epa_value"] = df["epa"].fillna(0)
    df["team_play_count"] = df.groupby(["season", "posteam"]).cumcount()
    df["def_play_count"] = df.groupby(["season", "defteam"]).cumcount()
    df["team_epa"] = (df.groupby(["season", "posteam"])["epa_value"].cumsum().shift(1).fillna(0) / df["team_play_count"].replace(0, 1))
    df["def_epa"] = (df.groupby(["season", "defteam"])["epa_value"].cumsum().shift(1).fillna(0) / df["def_play_count"].replace(0, 1))
    df["epa_diff"] = df["team_epa"] - df["def_epa"]
    df["team_recent_epa"] = df.groupby(["season", "posteam"])["epa_value"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["def_recent_epa"] = df.groupby(["season", "defteam"])["epa_value"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    return df


def build_wp_models(df, model_name):
    """
    Creates models that predict the post-play win probabilities for both attempt outcomes (success/failure).
    """
    features = [
        "yardline",
        "quarter",
        "seconds_remaining",
        "score_diff",
        "is_home",
        "is_goal_to_go",
        "is_red_zone",
        "team_recent_epa",
        "def_recent_epa",
        "epa_diff"
    ]

    df = create_wp_features(df)
    df = df.dropna(subset=features + ["wp_after"])
    # Get the resulting game state after the play (using state at start of next play) to base win probability prediction on
    next_features = [f"next_{feature}" for feature in features]
    df[next_features] = df.groupby("game_id")[features].shift(-1)
    df = df.dropna(subset=next_features + ["wp_after"])

    # Build separate models for if the attempt succeeds or fails
    models = {}
    for outcome_value, outcome_name in [(1, "success"), (0, "failure")]:
        matches = df[df["success"] == outcome_value]
        feature_df = matches[next_features].fillna(0)
        y = matches["wp_after"]

        X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        models[outcome_name] = model

        print(f"{model_name} WP {outcome_name} model R^2:", model.score(X_test, y_test))

    return models


def build_conversion_wp_models(conversion_df):
    return build_wp_models(conversion_df, "Conversion")


def build_field_goal_wp_models(field_goal_df):
    return build_wp_models(field_goal_df, "Field goal")


def build_punt_wp_models(punt_df):
    return build_wp_models(punt_df, "Punt")


def predict_wp_after_play(model, current_state):
    features = [
        "yardline",
        "quarter",
        "seconds_remaining",
        "score_diff",
        "is_home",
        "is_goal_to_go",
        "is_red_zone",
        "team_recent_epa",
        "def_recent_epa",
        "epa_diff"
    ]

    feature_df = current_state[features].copy()
    wp_after = model.predict(feature_df)[0]
    return np.clip(wp_after, 0, 1)


def swap_possession(state):
    """
    Updates the game state to reflect possession by the opposing team after certain play attempts and outcomes.
    """
    state = state.copy()
    state["score_diff"] = -state["score_diff"]
    state["is_home"] = 1 - state["is_home"]
    state[["team_recent_epa", "def_recent_epa"]] = state[["def_recent_epa", "team_recent_epa"]]
    state["epa_diff"] = -state["epa_diff"]
    state["is_goal_to_go"] = (state["yardline"] - state["yards_to_go"] == 0).astype(int)
    state["is_red_zone"] = (state["yardline"] <= 20).astype(int)
    return state


def simulate_post_conversion_state(current_state, success):
    """
    Simulates the post-play game state following a successful or failed conversion attempt for win probability prediction.
    """
    state = current_state.copy()
    state["seconds_remaining"] = max(state["seconds_remaining"] - 6, 0)
    possession_swapped = False

    if success:
        # If a successful conversion would mean a touchdown
        if state["is_goal_to_go"] == 1:
            state["score_diff"] += 7
            state["yardline"] = 75
            state = swap_possession(state)
            possession_swapped = True
        else:
            state["yardline"] = state["yardline"] - state["yards_to_go"]
            state["is_goal_to_go"] = (state["yardline"] - state["yards_to_go"] == 0).astype(int)
            state["is_red_zone"] = (state["yardline"] <= 20).astype(int)
    else:
        state["yardline"] = 100 - state["yardline"]
        state = swap_possession(state)
        possession_swapped = True

    return state, possession_swapped


def simulate_post_field_goal_state(current_state, success):
    """
    Simulates the post-play game state following a field goal make or miss for win probability prediction.
    """
    state = current_state.copy()
    state["seconds_remaining"] = max(state["seconds_remaining"] - 5, 0)
    if success:
        state["score_diff"] += 3
        state["yardline"] = 75
    else:
        state["yardline"] = 100 - (state["yardline"] + 7)

    state = swap_possession(state)
    return state, True


def simulate_post_punt_state(current_state, success):
    """
    Simulates the post-play game state following a "successful" or "failed" punt for win probability prediction.
    """
    state = current_state.copy()
    state["seconds_remaining"] = max(state["seconds_remaining"] - 6, 0)

    if success:
        state["yardline"] = random.randint(75, 90)
    else:
        state["yardline"] = random.randint(60, 70)

    state = swap_possession(state)
    return state, True


def predict_wp_outcomes(current_state, wp_models, simulator):
    """
    Returns the predicted post-play win probabilities for both attempt outcomes (success/failure) for expected win probability calculation.
    """
    success_state, success_possession_swapped = simulator(current_state, True)
    failure_state, failure_possession_swapped = simulator(current_state, False)

    wp_if_success = predict_wp_after_play(wp_models["success"], success_state)
    wp_if_failure = predict_wp_after_play(wp_models["failure"], failure_state)

    # If possession was swapped in the simulated states, get the WP for the team that made the 4th down decision (model will predict WP for the opposing team)
    if success_possession_swapped:
        wp_if_success = 1 - wp_if_success
    if failure_possession_swapped:
        wp_if_failure = 1 - wp_if_failure

    return wp_if_success, wp_if_failure


def predict_conversion_wp_outcomes(current_state, conversion_wp_models):
    return predict_wp_outcomes(current_state, conversion_wp_models, simulate_post_conversion_state)


def predict_field_goal_wp_outcomes(current_state, field_goal_wp_models):
    return predict_wp_outcomes(current_state, field_goal_wp_models, simulate_post_field_goal_state)


def predict_punt_wp_outcomes(current_state, punt_wp_models):
    return predict_wp_outcomes(current_state, punt_wp_models, simulate_post_punt_state)


def calculate_expected_wp(success_prob, wp_if_success, wp_if_failure):
    """
    Calculates the expected win probability of the play decision, taking into account the success probability of the play and the win probabilities if it succeeds/fails.
    """
    return (success_prob * wp_if_success + (1 - success_prob) * wp_if_failure)


def main():
    """
    Receives the success probability/win probability models and features for each play option, pickling them for prediction on user-entered game situations.
    """
    conversion_df, field_goal_df, punt_df = preprocess_data()

    conversion_success_model, conversion_success_features, conversion_model_df = build_conversion_success_model(conversion_df)
    with open("conversion_success_model.pkl", "wb") as f:
        pickle.dump({
                "model": conversion_success_model,
                "features": conversion_success_features
        }, f)

    conversion_wp_models = build_conversion_wp_models(conversion_model_df)
    with open("conversion_wp_models.pkl", "wb") as f:
        pickle.dump(conversion_wp_models, f)

    field_goal_success_model, field_goal_success_features, field_goal_model_df = build_field_goal_success_model(field_goal_df)
    with open("field_goal_success_model.pkl", "wb") as f:
        pickle.dump({
                "model": field_goal_success_model,
                "features": field_goal_success_features
        }, f)

    field_goal_wp_models = build_field_goal_wp_models(field_goal_model_df)
    with open("field_goal_wp_models.pkl", "wb") as f:
        pickle.dump(field_goal_wp_models, f)

    punt_success_model, punt_success_features, punt_model_df = build_punt_success_model(punt_df)
    with open("punt_success_model.pkl", "wb") as f:
        pickle.dump({
                "model": punt_success_model,
                "features": punt_success_features
        }, f)

    punt_wp_models = build_punt_wp_models(punt_model_df)
    with open("punt_wp_models.pkl", "wb") as f:
        pickle.dump(punt_wp_models, f)


if __name__ == "__main__":
    main()
