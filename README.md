Stock Price Prediction & Forecasting Dashboard

An end-to-end stock forecasting project built on live market data. It trains a Random Forest model, validates it honestly with walk-forward testing against a naive baseline, adds news sentiment as a feature, backtests a trading strategy on the predictions, and presents the results in both a Streamlit app and a Power BI dashboard.

Live Demo | Power BI Dashboard

Why this project

Most beginner stock predictors report a single train/test accuracy on shuffled data, which leaks future information and looks better than it really is. This project is built to avoid that:

Walk-forward validation respects time order, so the model never sees the future.
Naive baseline comparison shows whether the model actually beats "tomorrow's price equals today's price."
Backtesting checks whether predictions would have produced a useful strategy, not just a low error score.
Features
Live price data pulled with yfinance (no static dataset to go stale)
Feature engineering from historical price and volume data
Random Forest forecasting model
Walk-forward validation with model comparison against a naive baseline
News sentiment scoring using a transformer-based model, used as an additional feature
Strategy backtest on model predictions
Interactive Streamlit app for choosing a ticker and viewing forecasts
Results exported as CSV files that feed a Power BI dashboard
Tech stack
Area	Tools
Language	Python 3.13
Data	yfinance, pandas, NumPy
Modeling	scikit-learn (Random Forest)
Sentiment	Hugging Face Transformers, PyTorch
App	Streamlit
Dashboard	Power BI
Project structure
.
├── app.py                 # Streamlit app
├── requirements.txt
├── results/               # CSV outputs used by Power BI
├── powerbi/               # Power BI dashboard file / screenshots
└── README.md

Update this tree to match your actual folders and file names.

Results
Metric	Naive baseline	Random Forest	Random Forest + Sentiment
RMSE	YOUR_VALUE	YOUR_VALUE	YOUR_VALUE
MAE	YOUR_VALUE	YOUR_VALUE	YOUR_VALUE
Directional accuracy	YOUR_VALUE	YOUR_VALUE	YOUR_VALUE

Backtest summary: YOUR_STRATEGY_RETURN vs. buy-and-hold YOUR_BENCHMARK_RETURN over YOUR_PERIOD.

Add 1 or 2 screenshots here (forecast chart, Power BI dashboard):

![Forecast chart](images/forecast.png)
![Power BI dashboard](images/powerbi.png)
Getting started
1. Clone the repository
bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
2. Create a virtual environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the app
bash
streamlit run app.py
How it works
Data: Historical prices are downloaded live with yfinance for the selected ticker.
Features: Price- and volume-based features are engineered, and news sentiment scores are added.
Model: A Random Forest is trained and evaluated with walk-forward validation, so each prediction uses only past data.
Comparison: Results are compared with a naive baseline to confirm the model adds real value.
Backtest: Predictions drive a simple strategy that is tested against buy-and-hold.
Presentation: Results are shown in Streamlit and exported to CSV for the Power BI dashboard.
Limitations
Stock markets are noisy, and past performance does not guarantee future results.
News sentiment coverage depends on the headlines available for each ticker.
The backtest does not include transaction costs or slippage.
Disclaimer

This project is for educational purposes only and is not financial advice.

Future improvements
Add more models (for example gradient boosting, LSTM) to the comparison
Include transaction costs in the backtest
Support multi-ticker portfolio forecasting
Author

YOUR NAME Final-year Data Science / Computer Science student, BIHER LinkedIn | GitHub | Email
