# 📈 Stock Price Prediction Web App

This project is a **machine learning-powered web application** that predicts stock prices based on historical closing data. Built using Python, trained on real stock market data, and deployed with **Streamlit**, the app allows users to input any stock symbol (like `AAPL`, `GOOG`, `INFY`) and view predictions compared to actual values.

---

## 🔍 Features

- 📥 Fetches historical stock data using [`yfinance`](https://pypi.org/project/yfinance/)
- 📊 Visualizes price trends with a 100-day Moving Average
- 🧠 Loads a pre-trained model for prediction
- 🔮 Displays predicted vs actual prices graphically
- 🖥️ User-friendly interface powered by **Streamlit**

---

## 🛠️ Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| `Python` | Core programming |
| `yfinance` | Real-time stock data |
| `scikit-learn` | Data preprocessing (MinMaxScaler) |
| `pandas` & `numpy` | Data manipulation |
| `matplotlib` | Data visualization |
| `pickle` | Model loading |
| `Streamlit` | Web application deployment |
| `keras` / `tensorflow` | Model training (LSTM) *(used during training phase in `.ipynb`)* |

---

## 🧠 Model Details

The model used here is trained on the **closing prices** of stocks using a sequence of 100 previous time steps. It was built and trained using an LSTM-based architecture (in the Jupyter Notebook) and saved using `pickle`.

> Note: The training phase is done separately in the Jupyter notebook. The `.pkl` file is loaded in the Streamlit app for inference only.

---

## 🚀 How to Run the App

### 📦 1. Install dependencies
    pip install streamlit 
    pip install yfinance
    pip install pandas 
    pip install numpy 
    pip install matplotlib 
    pip install scikit-learn
    pip install keras
    pip install tensorflow

### ▶️ 2. Run the Streamlit app
    streamlit run app.py

---

### 📂 Project Structure


├── Stock_Price_Prediction.ipynb  # Training + EDA notebook

├── app.py                  # Streamlit Web App

├── model.pkl               # Trained ML model (pickled)
