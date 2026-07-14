import streamlit as st
import pickle
import pandas as pd
from engine import (
    predict_success_prob,
    predict_conversion_wp_outcomes,
    predict_field_goal_wp_outcomes,
    predict_punt_wp_outcomes,
    calculate_expected_wp
)


# Load all the success probability/win probability models for prediction
@st.cache_resource
def load_conversion_success_model():
    with open("conversion_success_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_conversion_wp_models():
    with open("conversion_wp_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_field_goal_success_model():
    with open("field_goal_success_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_field_goal_wp_models():
    with open("field_goal_wp_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_punt_success_model():
    with open("punt_success_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_punt_wp_models():
    with open("punt_wp_models.pkl", "rb") as f:
        return pickle.load(f)


conversion_success_model = load_conversion_success_model()
conversion_wp_models = load_conversion_wp_models()
field_goal_success_model = load_field_goal_success_model()
field_goal_wp_models = load_field_goal_wp_models()
punt_success_model = load_punt_success_model()
punt_wp_models = load_punt_wp_models()



team_pass_tendency = 1 - team_run_tendency
epa_diff = team_epa - def_epa
is_goal_to_go = (yardline - yards_to_go == 0).astype(int)
is_red_zone = (yardline <= 20).astype(int)
is_short_yardage = (yards_to_go <= 2).astype(int)
seconds_remaining = quarter_minutes_remaining * 60 + quarter_seconds_remaining
for i in range(quarter, 4):
    seconds_remaining += 900
score_diff = team_score - opp_score
def_wp = 1 - wp
wp_diff = wp - def_wp
kick_distance = yardline + 18

situation_data = pd.DataFrame({
    "yards_to_go": [yards_to_go],
    "yardline": [yardline],
    "team_run_tendency": [team_run_tendency],
    "team_pass_tendency": [team_pass_tendency],
    "epa_diff": [epa_diff],
    "team_recent_epa": [team_recent_epa],
    "def_recent_epa": [def_recent_epa],
    "is_goal_to_go": [is_goal_to_go],
    "is_red_zone": [is_red_zone],
    "is_short_yardage": [is_short_yardage],
    "quarter": [quarter],
    "seconds_remaining": [seconds_remaining],
    "score_diff": [score_diff],
    "is_home": [is_home],
    "wp": [wp],
    "def_wp": [def_wp],
    "wp_diff": [wp_diff],
    "team_recent_conversion_success": [team_recent_conversion_success],
    "def_recent_conversion_success": [def_recent_conversion_success],
    "kick_distance": [kick_distance],
    "temp": [temp],
    "wind": [wind]
})
