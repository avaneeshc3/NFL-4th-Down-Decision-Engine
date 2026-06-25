import nflreadpy
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle


def preprocess_data():
    # Load all available play-by-play historical data (from 1999)
    pbp_df = nflreadpy.load_pbp(seasons=True)
    pbp_df = pbp_df.to_pandas()
    pbp_df = pbp_df.drop_duplicates()
    pbp_df = pbp_df.dropna(subset=["down", "field_goal_attempt", "punt_attempt"])

    # Filter to respective play type attempts for each 4th down option - go for it, kick field goal, or punt
    conversion_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] != 1) & (pbp_df["punt_attempt"] != 1)]
    field_goal_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] == 1)]
    punt_df = pbp_df[(pbp_df["down"] == 4) & (pbp_df["punt_attempt"] == 1)]

    return conversion_df, field_goal_df, punt_df


def build_conversion_model(conversion_df):
    model_df = conversion_df.copy()
    model_df = model_df.dropna(subset=["season", "posteam", "defteam", "ydstogo", "yardline_100", "yards_gained"])

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
    model_df["success"] = ((model_df["yards_gained"].fillna(0) >= model_df["yards_to_go"]) | (model_df["touchdown"].fillna(0) == 1)).astype(int)

    features = ["yards_to_go", "yardline", "off_def_strength_diff", "team_run_tendency", "team_pass_tendency"]
    feature_df = model_df[features].copy().fillna(0)
    success = model_df["success"]

    # Build XGBClassifier model
    X_train, X_test, y_train, y_test = train_test_split(feature_df, success, test_size=0.25, stratify=success, random_state=42)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
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
        pickle.dump({"model": conversion_model, "features": conversion_features}, f)


if __name__ == "__main__":
    main()
