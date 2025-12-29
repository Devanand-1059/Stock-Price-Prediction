# 📈 Stock Price Prediction using Machine Learning (Streamlit App)

This project is an interactive **Stock Price Prediction Web Application** built using **Python, Machine Learning, and Streamlit**.  
It fetches real-time historical stock data, trains a machine learning model, predicts future prices, and exports analysis-ready datasets for visualization tools like **Power BI**.

 🚀 Features

- 🔍 Fetches real-time stock data using Yahoo Finance  
- 🤖 Uses **Random Forest Regressor** for price prediction  
- 📊 Visualizes historical vs predicted prices  
- 📈 Displays prediction summary with price change & percentage  
- 💾 Exports datasets for **Power BI / Excel analysis**  
- 🌙 Dark-mode optimized visualization  
- 🧑‍💻 Beginner-friendly and interactive UI  

🛠️ Technologies & Libraries Used

- Python  
- Streamlit  
- yfinance  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

## 📂 Project Structure

├── stockapp.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── AAPL_historical_data.csv # Sample exported historical dataset
├── AAPL_combined_analysis.csv # Sample combined historical + prediction data
├── AMZN_historical_data.csv # Sample exported historical dataset
├── AMZN_combined_analysis.csv # Sample combined historical + prediction data


 📊 Machine Learning Model

- **Algorithm Used:** Random Forest Regressor  
- **Input Feature:** Closing price  
- **Prediction Type:** Short-term future price prediction  
- **Evaluation Metrics:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

 📁 Data Export Details

The application exports two types of datasets:

 1️⃣ Historical Data
Includes:
- Date  
- Open, High, Low, Close, Volume  
- Moving Averages (7-day, 30-day)  
- Volatility  
- Daily Returns  

📄 File: `TICKER_historical_data.csv`

---

### 2️⃣ Combined Analysis Data
Includes:
- Historical prices  
- Predicted future prices  
- Data type (Historical / Predicted)  
- Ticker symbol  

📄 File: `TICKER_combined_analysis.csv`

These files can be directly imported into **Power BI or Excel** for dashboard creation.

▶️ How to Run the Project


Step 1: Clone the Repository
Run the following commands in your terminal:

git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction


Step 2: Install Dependencies
Make sure Python is installed, then install the required libraries:

pip install -r requirements.txt


Step 3: Run the Streamlit App
Start the application using:

streamlit run stockapp.py

---

🧪 Example Stock Tickers

AAPL

AMZN

MSFT

GOOGL

TSLA

INFY.NS


---

📌 Use Cases

Beginner Machine Learning Project

Data Science Portfolio Project

Power BI Dashboard Integration

Financial Data Analysis Practice

---

⚠️ Disclaimer

This project is for educational purposes only.
It should not be used for real-world financial or investment decisions.

---

👤 Author

Devanand S
Student | Aspiring Data Scientist & ML Engineer























