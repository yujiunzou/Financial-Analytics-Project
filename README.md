# 🔍 Financial Fraud Detector

**BA870 Financial Analytics – Team Project**  
members: Inchara Ashok, Yujiun Zou, Kara Liao
- FraudSight webapp : https://fraudsight.streamlit.app/
- Colab notebook:https://colab.research.google.com/drive/1_CwL3hADERWe9pJZ1_GnWR68GMpVwLWi?usp=sharing
## Overview
A Streamlit web app that detects potential financial statement fraud using:
- **Beneish M-Score** (rule-based, 8-variable linear model)
- **Random Forest classifier** (trained on WRDS Compustat data)

## Features
- 📈 Live public company analysis via Yahoo Finance (just enter a ticker)
- ✏️ Manual ratio input for private companies
- 📊 Radar chart visualization of financial ratios
- 🔑 Feature importance ranking from the ML model

## Project Structure
```
fraud_app/
├── app.py               # Main Streamlit application
├── train_model.py       # Offline model training script (run in Colab)
├── fraud_model.pkl      # Pre-trained Random Forest model
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup & Deployment

### 1. Train the Model (Offline in Google Colab)
Run `train_model.py` in Colab with your WRDS data to generate `fraud_model.pkl`.
Then download `fraud_model.pkl` and place it in this folder.

### 2. Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set `app.py` as the entry point
4. Click **Deploy**

## Data Sources
| Purpose | Source |
|---|---|
| Model Training | WRDS Compustat (offline only) |
| Live App Data | Yahoo Finance via `yfinance` (public domain) |

## Methodology
See the **About** tab inside the app for full methodology details.
