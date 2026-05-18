# ✈️ Turbofan Engine Predictive Maintenance

Predictive maintenance system for turbofan engines using the NASA C-MAPSS FD001 dataset.

This project predicts whether an engine is likely to fail within the next **30 operational cycles** using machine learning, temporal feature engineering, and sensor degradation analysis.

The project includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Comparison
- Leakage Detection
- Threshold Optimization
- Streamlit Deployment
- Deployment-Focused Evaluation

---

# 📌 Project Objective

Aircraft engine failures are extremely costly and potentially dangerous.

Traditional maintenance strategies are often:
- Reactive (after failure)
- Schedule-based (fixed intervals)

These approaches can lead to:
- Unnecessary maintenance
- Higher operational costs
- Unexpected failures

The goal of predictive maintenance is to detect degradation early enough to:
- Reduce downtime
- Improve safety
- Optimize maintenance scheduling

This project predicts:

- `0` → Healthy engine
- `1` → Engine likely to fail within 30 cycles

using engine sensor data from the NASA C-MAPSS turbofan simulation dataset.

---

# 📂 Dataset

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

**Subset Used:** `FD001`

FD001 contains:
- One operating condition
- One fault mode

making it suitable for building an initial predictive maintenance pipeline.

The dataset includes:
- Engine ID
- Cycle Number
- Operational Settings
- 21 Sensor Measurements

Each row represents one engine cycle.

---

# ⚙️ Project Workflow

## 1️⃣ Data Understanding and EDA

Performed:
- Engine Lifetime Analysis
- Missing Value Analysis
- Sensor Variance Analysis
- Correlation Analysis
- Sensor Degradation Trend Analysis

### Key Observations
- Several sensors were near-constant and removed
- Operational settings showed minimal variance in FD001
- Degradation patterns became more visible after ~100 cycles
- Engine lifetimes varied significantly across engines

---

## 2️⃣ Remaining Useful Life (RUL)

Calculated Remaining Useful Life:

```python
RUL = max_cycle - current_cycle
```

Converted into binary classification:

- `failure = 1` if `RUL ≤ 30`
- `failure = 0` otherwise

This created a class imbalance problem:
- Training failures ≈ 15%
- Test failures ≈ 2.5%

---

# 🛠️ Feature Engineering

Engine degradation is temporal.

Single-cycle sensor snapshots do not fully capture degradation behaviour.

To capture temporal patterns, I engineered:
- Rolling Means
- Rolling Standard Deviations
- Rate-of-Change Features

using rolling windows:
- 5 cycles
- 10 cycles
- 20 cycles

## 📈 Feature Engineering Results

| Feature Set | Recall | F1 Score |
|---|---|---|
| Raw Sensors | 0.869 | 0.878 |
| Rolling (5) | 0.866 | 0.893 |
| Rolling (5,10,20) | 0.877 | 0.910 |

### Observation
- Short windows captured local fluctuations
- Longer windows captured smoother degradation behaviour
- Combining multiple windows performed best

---

# 🚨 Leakage Discovery and Correction

One experiment introduced a feature called:

```python
cycle_norm
```

which represented normalized cycle position within engine lifetime.

This significantly improved performance.

However, I later realized this feature leaked future information because:
- It depended on the engine’s full lifetime
- That information would not be available during real deployment

After removing the feature:
- Performance decreased
- But the model became more realistic and deployment-safe

This became one of the most important lessons of the project:

> Unusually strong performance can sometimes indicate leakage rather than genuine predictive capability.

---

# 🤖 Models Evaluated

Models compared:
- Logistic Regression
- Logistic Regression (Balanced)
- Random Forest
- Gradient Boosting

## 📊 Cross-Validation Recall

| Model | Mean Recall |
|---|---|
| Logistic Regression (Balanced) | 0.971 |
| Random Forest | 0.912 |
| Gradient Boosting | 0.927 |

---

# 🎯 Threshold Optimization

The default threshold (`0.5`) was not ideal for predictive maintenance because:
- Missing failures is more costly than false alarms

Threshold tuning improved recall:

| Threshold | Precision | Recall |
|---|---|---|
| 0.50 | 0.947 | 0.873 |
| 0.35 | 0.883 | 0.921 |

### Final Selected Threshold
`0.35`

---

# 🧪 Final Test Performance

## Validation Performance

| Metric | Score |
|---|---|
| Recall | 0.921 |
| Precision | 0.883 |
| F1 Score | 0.901 |

## Test Performance

| Metric | Score |
|---|---|
| Recall | 0.663 |
| Precision | 0.794 |
| F1 Score | 0.720 |

---

# ⚠️ Important Observation

Validation performance dropped significantly on the test set.

### Main Reasons
- Severe class distribution shift
- Threshold sensitivity
- Deployment realism
- Differences in degradation behaviour

This reinforced an important machine learning lesson:

> Strong validation metrics alone are not enough.

Real deployment performance depends heavily on:
- Realistic evaluation
- Distribution similarity
- Deployment consistency

---

# 🔍 Feature Importance

Most important features:
- Rolling temperature statistics
- Rotational speed trends
- Compressor outlet temperature behaviour

The model relied much more on:
- Temporal degradation trends

than:
- Isolated sensor snapshots

This aligned well with turbofan degradation physics.

---

# 🚀 Streamlit Deployment

The project includes a deployed Streamlit application for interactive inference.

### Features
- Multi-cycle engine input
- Rolling feature generation
- Failure probability prediction
- Risk interpretation
- Deployment-aware feature handling

## ⚠️ Deployment Limitation

The model was trained using temporal rolling features.

This means prediction quality depends heavily on:
- Recent engine history
- Temporal degradation behaviour

Single-cycle snapshots alone are often insufficient for reliable predictive maintenance inference.

This became an important deployment lesson during the project.

---

# 🧰 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

# 📁 Repository Structure

```text
├── predictive_maintenance.ipynb
├── app.py
├── engine_failure_model.pkl
├── feature_names.json
├── requirements.txt
└── README.md
```

---

# 🔮 Future Improvements

Potential future improvements include:
- LSTM / GRU sequence models
- Transformer architectures
- Probability calibration
- Optuna / Bayesian optimization
- Deployment using streaming sensor data
- Evaluation on FD002–FD004 datasets
- Online monitoring pipelines
- Integration with maintenance logs and operational metadata

---

# 📚 Key Lessons From The Project

This project evolved beyond a standard classification task.

Major lessons included:
- Leakage prevention
- Threshold optimization
- Deployment consistency
- Evaluation realism
- Temporal feature engineering
- Handling distribution shift

One of the most important realizations was that:
- Training pipelines
- Validation pipelines
- Deployment pipelines

must remain consistent for machine learning systems to generalize reliably.

---

# 💡 What This Project Demonstrates

This project demonstrates:
- End-to-end machine learning workflow
- Predictive maintenance modeling
- Temporal feature engineering
- Leakage detection and correction
- Threshold optimization
- Cross-validation with grouped data
- Deployment using Streamlit
- Realistic evaluation under distribution shift
- MLOps-aware deployment reasoning

---

# 🌐 Streamlit App

https://turbofan-predictive-maintenance-jcbxyf2thw86qvrdqbcczt.streamlit.app/

---

# 🔗 GitHub Repository

https://github.com/sushantpandey06/turbofan-predictive-maintenance

---

