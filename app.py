import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Turbofan Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# ============================================================
# Load Model and Features
# ============================================================

model = joblib.load("engine_failure_model.pkl")

with open("feature_names.json") as f:
    feature_names = json.load(f)

# ============================================================
# Sensor List
# ============================================================

sensor_list = [
    'sensor_2',
    'sensor_3',
    'sensor_4',
    'sensor_7',
    'sensor_9',
    'sensor_11',
    'sensor_12',
    'sensor_14',
    'sensor_17',
    'sensor_20',
    'sensor_21'
]

# ============================================================
# App Title
# ============================================================

st.title("✈️ Turbofan Engine Predictive Maintenance")

st.markdown("""
This application predicts whether a turbofan engine is likely to fail within the next **30 cycles** using sensor-based predictive maintenance modeling.

The model was trained on the NASA **C-MAPSS FD001** dataset using:
- rolling temporal features,
- degradation trends,
- and Gradient Boosting classification.
""")

# ============================================================
# Deployment Note
# ============================================================

st.info("""
This model performs best when multiple recent engine cycles are provided.

The application computes rolling statistics dynamically from recent sensor history to better match the feature engineering pipeline used during training.
""")

# ============================================================
# Sidebar Configuration
# ============================================================

st.sidebar.header("Input Configuration")

num_cycles = st.sidebar.slider(
    "Number of Recent Cycles",
    min_value=5,
    max_value=20,
    value=10
)

st.sidebar.markdown("""
Enter recent sensor values for the engine.

Rows represent sequential engine cycles.
""")

# ============================================================
# Default Example Values
# ============================================================

default_values = {
    'sensor_2': 641.82,
    'sensor_3': 1589.70,
    'sensor_4': 1400.60,
    'sensor_7': 554.36,
    'sensor_9': 9046.19,
    'sensor_11': 47.47,
    'sensor_12': 521.66,
    'sensor_14': 8125.55,
    'sensor_17': 392,
    'sensor_20': 38.86,
    'sensor_21': 23.37
}

# ============================================================
# Multi-Cycle Input Table
# ============================================================

st.subheader("Recent Engine Sensor History")

input_df = pd.DataFrame()

for sensor in sensor_list:

    input_df[sensor] = [
        default_values[sensor]
        for _ in range(num_cycles)
    ]

edited_df = st.data_editor(
    input_df,
    num_rows="fixed",
    use_container_width=True
)

# ============================================================
# Feature Engineering Function
# ============================================================

def create_features(df):

    feature_dict = {}

    for sensor in sensor_list:

        values = df[sensor].values

        # Rolling Means
        for w in [5, 10, 20]:

            window = min(w, len(values))

            roll_mean = (
                pd.Series(values)
                .rolling(window)
                .mean()
                .iloc[-1]
            )

            roll_std = (
                pd.Series(values)
                .rolling(window)
                .std()
                .fillna(0)
                .iloc[-1]
            )

            feature_dict[
                f"{sensor}_rollmean_{w}"
            ] = roll_mean

            feature_dict[
                f"{sensor}_rollstd_{w}"
            ] = roll_std

        # Rate of Change
        if len(values) > 1:

            roc = values[-1] - values[-2]

        else:

            roc = 0

        feature_dict[f"{sensor}_roc"] = roc

    return pd.DataFrame([feature_dict])

# ============================================================
# Prediction
# ============================================================

if st.button("Predict Failure Risk"):

    try:

        # Create features
        features_df = create_features(edited_df)

        # Ensure feature ordering consistency
        features_df = features_df.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Predict
        probability = model.predict_proba(
            features_df
        )[0][1]

        THRESHOLD = 0.35

        # ====================================================
        # Results
        # ====================================================

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Failure Probability",
                f"{probability:.2%}"
            )

        with col2:

            if probability >= 0.7:

                risk_level = "CRITICAL"

            elif probability >= THRESHOLD:

                risk_level = "WARNING"

            else:

                risk_level = "SAFE"

            st.metric(
                "Risk Level",
                risk_level
            )

        # ====================================================
        # Risk Messages
        # ====================================================

        if probability >= 0.7:

            st.error("""
            ⚠️ CRITICAL RISK

            The engine shows strong degradation patterns consistent with engines approaching failure.

            Immediate inspection and maintenance are recommended.
            """)

        elif probability >= THRESHOLD:

            st.warning("""
            ⚠️ WARNING

            The engine shows noticeable degradation behaviour.

            Preventive maintenance inspection is recommended soon.
            """)

        else:

            st.success("""
            ✅ SAFE

            Current sensor behaviour does not strongly indicate imminent failure.

            Continue regular monitoring.
            """)

        # ====================================================
        # Probability Bar
        # ====================================================

        st.progress(float(min(probability, 1.0)))

        # ====================================================
        # Technical Explanation
        # ====================================================

        with st.expander("Technical Details"):

            st.markdown(f"""
            ### Model Information

            - Model: Gradient Boosting Classifier
            - Prediction Threshold: {THRESHOLD}
            - Failure Definition: Failure expected within 30 cycles
            - Dataset: NASA C-MAPSS FD001

            ### Feature Engineering

            The model uses:
            - rolling means,
            - rolling standard deviations,
            - and sensor rate-of-change features

            computed from recent engine cycles.

            ### Important Note

            Predictions are most reliable when recent engine history is available.

            Predictive maintenance models depend heavily on temporal degradation behaviour rather than isolated sensor snapshots.
            """)

    except Exception as e:

        st.error(f"Prediction Error: {e}")