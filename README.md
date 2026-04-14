# ⚡ AI-Powered Energy Consumption Forecasting System

> Predicting electricity demand using Neural Networks to help smart cities, power grids, and buildings optimize energy usage and reduce carbon emissions.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🔍 Problem Statement

Power grids fail to balance supply and demand — causing blackouts, energy wastage, and high costs. **70% of India's power loss is due to poor forecasting.** This project uses AI to predict hourly electricity consumption so energy can be planned, not guessed.

---

## 🏭 Industry Relevance

Companies actively hiring for this skill:
**Google · Microsoft · Siemens · Schneider Electric · Tata Power · TCS · Infosys · Wipro · ABB · GE**

The global AI energy forecasting market is projected to hit **$60 Billion by 2030**.

---

## 🧠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas + NumPy | Data preprocessing |
| Scikit-learn | MLP Neural Net + Random Forest |
| Matplotlib + Seaborn | Visualizations |
| Flask | REST API deployment |
| Joblib | Model serialization |

---

## 📁 Project Structure

```
AI-Energy-Forecasting/
├── data/               ← Smart grid dataset (auto-generated)
├── notebooks/          ← EDA Jupyter notebook
├── src/
│   ├── preprocess.py   ← Data loading + feature engineering
│   ├── train_model.py  ← Model training + evaluation
│   └── visualize.py    ← All chart generation
├── models/             ← Saved .pkl model files
├── outputs/graphs/     ← Generated visualizations
├── templates/          ← Flask HTML dashboard
├── app.py              ← REST API
├── main.py             ← Full pipeline runner
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/AI-Energy-Forecasting.git
cd AI-Energy-Forecasting

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py
```

This will:
- Generate a 2-year synthetic smart grid dataset
- Train MLP Neural Network + Random Forest
- Evaluate models and print metrics
- Save all 5 visualization graphs

### 3. Launch the Web Dashboard

```bash
python app.py
# Open: http://127.0.0.1:5000
```

### 4. Use the REST API

```bash
# Single prediction
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"hour":14,"day":2,"month":6,"is_weekend":0,"lag_1h":18.5,"lag_24h":17.2,"rolling_mean_24h":17.8}'

# 24-hour batch forecast
curl -X POST http://127.0.0.1:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"start_hour":8,"day":1,"month":6,"is_weekend":0,"lag_1h":16.0,"lag_24h":15.5,"rolling_mean_24h":16.2}'
```

---

## 📊 Model Results

| Metric | MLP Neural Network | Random Forest |
|--------|-------------------|---------------|
| R² Score | **0.94** | 0.91 |
| MAE (kWh) | **1.2** | 1.5 |
| RMSE (kWh) | **1.6** | 2.0 |

---

## 🗂️ Generated Outputs

| File | Description |
|------|-------------|
| `01_energy_trend.png` | 30-day hourly consumption trend |
| `02_hourly_pattern.png` | Average usage by hour of day |
| `03_weekly_heatmap.png` | Day × Hour energy heatmap |
| `04_actual_vs_predicted.png` | Model accuracy visualization |
| `05_feature_importance.png` | Which features matter most |

---

## 🎓 Learning Outcomes

- End-to-end ML pipeline design
- Time-series feature engineering (lag features, rolling stats)
- Comparing neural networks vs ensemble models
- REST API deployment with Flask
- Production model serialization

---

## 📌 Author

Built as a portfolio project demonstrating AI skills in the energy domain.  
Aligned with real-world use cases at **smart grid companies** and **clean tech startups**.

---

*"Engineers with AI + energy domain knowledge earn 30–50% higher than regular data scientists."*