import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import os

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction")

# Input section
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, INFY.NS, AMZN, GOOGL, MSFT, TSLA)", "AAPL")
days_to_predict = st.slider("Days to Predict into the Future", 1, 15, 5)

# Data export options
st.subheader("📁 Data Export Options")
col1, col2 = st.columns(2)
with col1:
    export_raw_data = st.checkbox("Export Raw Historical Data", value=True)
with col2:
    export_predictions = st.checkbox("Export Predictions Data", value=True)

if st.button("Predict"):
    try:
        # Step 1: Download Data
        st.info("Downloading stock data...")
        df = yf.download(ticker, period='2y', interval='1d')
        if df.empty:
            st.error("Failed to fetch data. Check the stock ticker.")
            st.stop()

        # Save raw historical data for Power BI
        if export_raw_data:
            raw_data = df.copy()
            raw_data.reset_index(inplace=True)
            raw_data['Ticker'] = ticker
            # Format date as string for better Excel compatibility
            raw_data['Date'] = pd.to_datetime(raw_data['Date']).dt.strftime('%Y-%m-%d')
            
            # Add technical indicators for better Power BI analysis
            raw_data['MA_7'] = raw_data['Close'].rolling(window=7).mean()
            raw_data['MA_30'] = raw_data['Close'].rolling(window=30).mean()
            raw_data['Volatility'] = raw_data['Close'].rolling(window=30).std()
            raw_data['Daily_Return'] = raw_data['Close'].pct_change()
            
            raw_filename = f"{ticker}_historical_data.csv"
            raw_data.to_csv(raw_filename, index=False)
            st.success(f"✅ Raw historical data saved as: {raw_filename}")

        # Prepare data for prediction model
        df = df[['Close']].dropna()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        if len(df) < days_to_predict + 10:
            st.error("Not enough data for prediction. Try fewer days.")
            st.stop()

        # Step 2: Train the Model
        X = df[['Close']].values[:-days_to_predict]
        y = df['Target'].values[:-days_to_predict]
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)

        # Step 3: Evaluate the Model
        test_X = df[['Close']].values[-days_to_predict:]
        test_y = df['Target'].values[-days_to_predict:]

        # Remove NaNs
        valid_indices = ~np.isnan(test_y)
        test_X = test_X[valid_indices]
        test_y = test_y[valid_indices]

        if len(test_X) > 0:
            preds = model.predict(test_X)
            mse = mean_squared_error(test_y, preds)
            rmse = np.sqrt(mse)

            st.subheader("📉 Model Evaluation")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Step 4: Predict Future Prices
        future_predictions = []
        last_input = df['Close'].values[-1]

        for i in range(days_to_predict):
            next_pred = model.predict(np.array([[last_input]]).reshape(1,-1))
            future_predictions.append(next_pred[0])
            last_input = next_pred[0]

        # Step 5: Create DataFrame for Future Predictions
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
        pred_df.set_index('Date', inplace=True)

        st.subheader("📅 Predicted Prices Table")
        st.dataframe(pred_df)

        # Step 6: Save combined dataset for Power BI
        if export_predictions:
            # Prepare historical data
            historical_data = df[['Close']].copy()
            historical_data.reset_index(inplace=True)
            historical_data['Type'] = 'Historical'
            historical_data['Ticker'] = ticker
            # Format date as string for better Excel compatibility
            historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.strftime('%Y-%m-%d')
            historical_data.rename(columns={'Close': 'Price'}, inplace=True)

            # Prepare prediction data
            prediction_data = pred_df.copy()
            prediction_data.reset_index(inplace=True)
            prediction_data['Type'] = 'Predicted'
            prediction_data['Ticker'] = ticker
            # Format date as string for better Excel compatibility
            prediction_data['Date'] = pd.to_datetime(prediction_data['Date']).dt.strftime('%Y-%m-%d')
            prediction_data.rename(columns={'Predicted Price': 'Price'}, inplace=True)

            # Combine both datasets
            combined_df = pd.concat([
                historical_data[['Date', 'Price', 'Type', 'Ticker']],
                prediction_data[['Date', 'Price', 'Type', 'Ticker']]
            ], ignore_index=True)

            # Sort by date
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            combined_filename = f"{ticker}_combined_analysis.csv"
            combined_df.to_csv(combined_filename, index=False)
            st.success(f"✅ Combined analysis data saved as: {combined_filename}")

        # Step 7: Plot Graph - Dark Mode Optimized
        st.subheader("📊 Historical vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Background Colors
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        # Plot Lines
        ax.plot(df.index, df['Close'], label='Historical', color='yellow', linewidth=2)
        ax.plot(pred_df.index, pred_df['Predicted Price'], label='Predicted', color='white', linewidth=2, marker='o')

        # Prediction Start Marker
        ax.axvline(df.index[-1], color='red', linestyle='--', label='Prediction Start')

        # Labels and Titles
        ax.set_title(f"{ticker} Stock Prediction", fontsize=14, color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Price", color='white')

        # Tick Colors
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')

        # Grid and Legend
        ax.grid(True, linestyle='--', alpha=0.3)
        legend = ax.legend(facecolor="#1e1e1e",edgecolor="white")
        for text in legend.get_texts():
            text.set_color("yellow")

        # Show Plot
        st.pyplot(fig)

        # Step 8: Summary Metrics
        current_price = float(df['Close'].values[-1])
        predicted_price = float(future_predictions[-1])
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100

        st.subheader("📈 Prediction Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric(f"Predicted in {days_to_predict} Days", f"${predicted_price:.2f}", f"{change:+.2f}")
        col3.metric("Predicted Change (%)", f"{change_pct:+.2f}%", f"${change:+.2f}")

        # Step 9: File download section
        st.subheader("💾 Download Files")
        files_created = []
        
        if export_raw_data and os.path.exists(f"{ticker}_historical_data.csv"):
            files_created.append(f"{ticker}_historical_data.csv")
            
        if export_predictions and os.path.exists(f"{ticker}_combined_analysis.csv"):
            files_created.append(f"{ticker}_combined_analysis.csv")
        
        if files_created:
            st.info(f"📁 Files created in your working directory:")
            for file in files_created:
                st.write(f"• {file}")
        
        # Display sample of the data for verification
        if export_predictions and os.path.exists(f"{ticker}_combined_analysis.csv"):
            st.subheader("🔍 Sample of Combined Dataset")
            sample_data = pd.read_csv(f"{ticker}_combined_analysis.csv")
            st.dataframe(sample_data.head(10))
            st.write(f"Total records: {len(sample_data)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug info:", str(e))

