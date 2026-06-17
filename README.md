# ⚙️ Predictive Maintenance — Remaining Useful Life Prediction

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/AN1-M/predictive-maintenance)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)](https://streamlit.io)

> Predicts how many operational cycles remain before a turbofan engine fails — enabling proactive maintenance scheduling and reducing unplanned industrial downtime.

**[🚀 Try the Live Dashboard →](https://huggingface.co/spaces/AN1-M/predictive-maintenance)**

---

## 📌 Business Context

Unplanned equipment failure costs industrial manufacturers an estimated **$50 billion annually** in downtime, emergency repairs, and lost production. Traditional maintenance is either reactive (fix after failure) or schedule-based (fix at fixed intervals regardless of actual condition).

**Predictive Maintenance** changes this: by monitoring real-time sensor data and predicting remaining useful life (RUL), maintenance teams can intervene *exactly when needed* — before failure, but not wastefully early.

This project demonstrates a full ML pipeline for RUL prediction directly applicable to Industry 4.0 use cases at companies like **Bosch, Festo, Mercedes-Benz, and Siemens**.

---

## 📊 Results

| Model | RMSE (cycles) | MAE (cycles) | NASA Score |
|-------|:---:|:---:|:---:|
| XGBoost (Baseline) | 19.39 | 13.97 | 752.65 |
| LSTM | 15.35 | 10.77 | 436.55 |
| **Transformer** | **14.97** | **10.94** | **343.87** |

> NASA Score is an asymmetric penalty metric — **late predictions are penalized more heavily than early ones**, reflecting the real-world cost asymmetry: predicting failure too late risks catastrophic breakdown, while predicting too early means unnecessary (but safe) maintenance.

---

## 🗂️ Dataset

**NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)**

- 4 subsets (FD001–FD004) with varying operating conditions and fault modes
- Each row = one engine at one time cycle with 21 sensor readings
- Training data runs until engine failure; test data is cut off before failure
- **This project uses FD001**: single operating condition, single fault mode

| Split | Engines | Rows |
|-------|---------|------|
| Train | 100 | 20,631 |
| Test | 100 | 13,096 |

---

## 🏗️ Project Architecture

```
predictive-maintenance/
├── data/
│   ├── raw/CMaps/          ← NASA CMAPSS dataset
│   └── processed/          ← feature-engineered data + results
├── notebooks/
│   ├── 01_eda.ipynb        ← sensor variance analysis
│   └── 02_train_evaluate.ipynb ← model training + comparison
├── src/
│   ├── data_loader.py      ← load CMAPSS, compute RUL
│   ├── features.py         ← feature engineering pipeline
│   ├── evaluate.py         ← RMSE, MAE, NASA scoring function
│   └── models/
│       ├── xgboost_model.py
│       ├── lstm_model.py
│       └── transformer_model.py
├── app/
│   └── streamlit_app.py    ← local dashboard
├── app.py                  ← HF Spaces entry point
└── requirements.txt
```

---

## 🔬 Methodology

### 1. Feature Engineering

Raw sensor readings are noisy — a single spike doesn't indicate failure. The pipeline extracts meaningful degradation signals:

- **Sensor filtering**: 7 zero-variance sensors dropped (std < 0.01), identified via automated variance analysis
- **RUL clipping**: Target clipped at 125 cycles — models focus on the degradation window, not healthy operation
- **Rolling features**: Mean and std over a 10-cycle window per sensor per engine (captures trend and volatility)
- **Lag features**: Sensor readings from 1, 3, and 5 cycles ago (captures rate of change)
- **Normalization**: MinMax scaling fitted on train only — no data leakage

Final feature count: **81 features** (from 21 raw sensors)

### 2. Models

**XGBoost (Baseline)**
- Gradient boosted trees on tabular features
- No sequence awareness — treats each cycle independently
- Fast to train, highly explainable (feature importance)
- Establishes the performance floor

**LSTM**
- Processes sequences of 30 cycles per engine
- Hidden size: 64, 2 stacked layers, dropout: 0.2
- Captures temporal degradation patterns XGBoost misses
- Training: Adam optimizer, MSELoss, 50 epochs

**Transformer**
- Multi-head self-attention over 30-cycle sequences
- Positional encoding injects cycle order information
- Global average pooling across all timesteps (vs. LSTM's last-step only)
- Gradient clipping + ReduceLROnPlateau scheduler
- Training: 60 epochs

### 3. Evaluation

Three complementary metrics:
- **RMSE**: Standard regression error, sensitive to outliers
- **MAE**: Average absolute error in cycles
- **NASA Score**: Asymmetric penalty (late predictions cost more) — the metric that actually matters in production

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/AniruddhMukherjee/predictive-maintenance
cd predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Download NASA CMAPSS dataset
kaggle datasets download -d behrad3d/nasa-cmaps -p data/raw/ --unzip

# Train all models
python src/train.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 📈 Dashboard Features

- **Model Comparison**: Side-by-side RMSE, MAE, NASA Score table + bar charts
- **Engine Deep Dive**: Select any test engine, see predictions from all 3 models vs actual RUL
- **Sensor Degradation Plot**: Visualize how key sensors evolve over the engine's lifetime
- **Prediction vs Actual Scatter**: See where models perform well and where they struggle

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| ML/DL | PyTorch, XGBoost, Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Hugging Face Spaces |
| Development | GitHub Codespaces |

---

## 👤 Author

**Aniruddh Mukherjee**
MSc INFOTECH — Universität Stuttgart (2026)
Published researcher (Springer) | Ex-Linde Engineering

[![GitHub](https://img.shields.io/badge/GitHub-AniruddhMukherjee-black)](https://github.com/AniruddhMukherjee)
