# Sentiment Dashboard

A lightweight Streamlit app that pulls **live news headlines via RSS** (BBC, Reuters, Guardian), runs **VADER sentiment**, and shows a quick **distribution + table**.  
*Round 1 focuses on a working MVP with no API keys. Round 2 will add time series, model upgrades, and export.*

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
