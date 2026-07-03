# =========================================================
# Crop Recommendation - Streamlit App
# =========================================================
# How to run:
#   1. Place these files in the SAME folder as this script:
#        - best_crop_model.pkl
#        - scaler.pkl
#        - feature_columns.pkl
#        - label_encoder.pkl
#        - best_model_name.pkl
#      (these get created after running crop_recommendation_model.py)
#   2. In terminal:  streamlit run streamlit_app.py
# =========================================================

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

st.title("🌾 Crop Recommendation System")
st.write(
    "Apni soil aur weather ki values daalo, model batayega kaunsi fasal "
    "sabse best rahegi."
)

# ---------------------------------------------------------
# LOAD MODEL + PREPROCESSING OBJECTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_crop_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    best_model_name = joblib.load("best_model_name.pkl")
    return model, scaler, feature_columns, label_encoder, best_model_name


try:
    model, scaler, feature_columns, label_encoder, best_model_name = load_artifacts()
    st.caption(f"Using model: **{best_model_name}**")
except FileNotFoundError:
    st.error(
        "Model files nahi mile. Pehle crop_recommendation_model.py run karo, "
        "phir generated .pkl files ko is app ke folder mein daalo."
    )
    st.stop()

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.subheader("Input Values")

# Default full feature set (in case Phosphorus wasn't dropped for your data)
all_possible_features = {
    "Nitrogen": (0, 140, 50),
    "Phosphorus": (5, 145, 50),
    "Potassium": (5, 205, 50),
    "Temperature": (8.0, 45.0, 25.0),
    "Humidity": (14.0, 100.0, 70.0),
    "pH_Value": (3.5, 10.0, 6.5),
    "Rainfall": (20.0, 300.0, 100.0),
}

col1, col2 = st.columns(2)
user_input = {}

for i, feature in enumerate(feature_columns):
    min_v, max_v, default_v = all_possible_features.get(feature, (0.0, 500.0, 50.0))
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        user_input[feature] = st.number_input(
            feature,
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(default_v),
        )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("Predict Crop 🌱"):
    input_df = pd.DataFrame([user_input])[feature_columns]
    input_scaled = scaler.transform(input_df.values)

    prediction = model.predict(input_scaled)[0]

    # If the best model was XGBoost, prediction is a label-encoded int
    if best_model_name == "xgboost":
        prediction = label_encoder.inverse_transform([prediction])[0]

    st.success(f"Recommended Crop: **{prediction}**")
