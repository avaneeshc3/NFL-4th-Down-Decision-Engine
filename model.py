import nflreadpy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

def preprocess_data():
    # Load all available play-by-play historical data (from 1999)
    pbp_df = nflreadpy.load_pbp(seasons=True)
    pbp_df = pbp_df.drop_duplicates()
    pbp_df = pbp_df.dropna(subset=["down", "field_goal_attempt", "punt_attempt"])
    
    # Filter to respective play type attempts for each 4th down option - go for it, kick field goal, or punt
    conversion_df = pbp_df.filter((pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] != 1) & (pbp_df["punt_attempt"] != 1))
    field_goal_df = pbp_df.filter((pbp_df["down"] == 4) & (pbp_df["field_goal_attempt"] == 1))
    punt_df = pbp_df.filter((pbp_df["down"] == 4) & (pbp_df["punt_attempt"] == 1))


def main():
    preprocess_data()

if __name__ == "__main__":
    main()