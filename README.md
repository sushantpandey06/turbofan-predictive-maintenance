# Turbofan Engine Failure Prediction

Predicts whether a turbofan engine will fail within the next 30 cycles using sensor data.

Dataset: NASA C-MAPSS FD001 — 100 engines, 21 sensors, each engine run until failure.

Dataset link - https://drive.google.com/drive/folders/1-PSZjjly7cU5FBWGjEEA5FmUpQJ52YeL?usp=sharing

## What I did

- Removed 10 sensors that had near-zero variance (basically constant readings)
- Created rolling mean, rolling std, and rate-of-change features at windows 5, 10, and 20 cycles for each of the remaining 11 sensors. That gave 77 features total
- Tried Logistic Regression, Random Forest, and Gradient Boosting. Picked GB
- Lowered the prediction threshold from 0.5 to 0.35 because missing a failure is worse than a false alarm
- Deployed on Streamlit so you can enter sensor values and get a prediction

## The leakage problem

Early on I created a feature called cycle_norm (current cycle / max cycle). It got 91% feature importance and the model gave recall of 1.0 on test — which seemed too good. Turned out cycle_norm uses the engine's total lifetime, which you only know after it's already failed. That's future information. I removed it. Recall dropped but the model became honest.

## Results

Validation: 92.1% recall, 88.3% precision
Test: 66.3% recall — dropped because test data has 2.5% failure rate vs 15% in training

The top features were sensor_4 (LPT outlet temperature) and sensor_11 (corrected core speed), both connected to how turbofan engines actually degrade.

## How to run

Notebook: open `Turbofan_predictive_maintenance.ipynb` in Colab or Jupyter

Streamlit app:
```
pip install -r requirements.txt
streamlit run app.py
```

App link: https://turbofan-predictive-maintenance-jcbxyf2thw86qvrdqbcczt.streamlit.app/

## What I'd do differently

- Try LSTM since this is sequential data
- Recalibrate the threshold on data that matches deployment class distribution
- Test on FD002-FD004 which have multiple operating conditions
