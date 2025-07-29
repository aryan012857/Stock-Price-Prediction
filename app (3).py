import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

import matplotlib.pyplot as plt

# App Title
st.title('📈 Stock Price Predictor')

# Dropdown Stock Symbol Input
selected_stock = st.selectbox(
    '📌 Select a Stock Symbol',
    ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'META', 'INFY', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
)

if st.button("Show Prediction"):
    if selected_stock:
        # Fetch data
        start = '2012-01-01'
        end = '2022-12-12'

        df = yf.download(selected_stock, start=start, end=end)

        if not df.empty:
            # Show data
            st.subheader('📊 Historical Stock Data')
            st.dataframe(df.tail())

            # Split into train and test
            train_data = pd.DataFrame(df['Close'][0:int(len(df) * 0.80)])
            test_data = pd.DataFrame(df['Close'][int(len(df) * 0.80):])

            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))

            past_100_days = train_data.tail(100)
            final_test_data = pd.concat([past_100_days, test_data], ignore_index=True)
            scaled_data = scaler.fit_transform(final_test_data)

            # MA100 Plot
            st.subheader('📉 Close Price vs MA100')
            ma100 = df['Close'].rolling(100).mean()
            fig1 = plt.figure(figsize=(10, 4))
            plt.plot(ma100, 'r', label='MA100')
            plt.plot(df['Close'], 'g', label='Close Price')
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Price")
            st.pyplot(fig1)

            # Prepare test features
            x_test = []
            y_test = []

            for i in range(100, scaled_data.shape[0]):
                x_test.append(scaled_data[i - 100:i])
                y_test.append(scaled_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Predict
            y_pred = model.predict(x_test)

            # Rescale back to original
            scale_factor = 1 / scaler.scale_[0]
            y_pred = y_pred * scale_factor
            y_test = y_test * scale_factor

            # Plot prediction vs actual
            st.subheader('🔮 Predicted vs Actual Prices')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual Price', color='green')
            plt.plot(y_pred, label='Predicted Price', color='red')
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.legend()
            st.pyplot(fig2)

        else:
            st.error("❌ No data found for this stock symbol.")
    else:
        st.info("💡 Please enter a valid stock symbol to get started.")