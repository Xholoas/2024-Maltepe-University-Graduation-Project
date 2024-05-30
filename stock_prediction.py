import os
import random

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from datetime import date
import streamlit as st
import yfinance as yf
from yahoo_fin.stock_info import tickers_sp500

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go

# Get S&P 500 stock symbols
sp500_symbols = tickers_sp500()

def load_custom_css():
    with open("custom.css", "r") as file:
        css = file.read()
        return css

@st.cache_data
def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    return scaled_data, scaler

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return np.round(mae,3), np.round(mse,3), np.round(rmse,3)

def suggest_symbol():
     st.session_state.selected_stock = random.choice(sp500_symbols)

def get_company_info(symbol, length=250):
    company = yf.Ticker(symbol)
    info = company.info
    company_info = [
        info.get("longName", "Not available"),
        info.get("website", "Not available"),
        info.get("industry", "Not available"),
        info.get("sector", "Not available"),
        info.get("longBusinessSummary", "Not available")[:length]
    ]
    return company_info

@st.cache_data
def train_lstm_model(X_train, y_train, epochs, batch_size):
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

@st.cache_data
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_historical_lstm(model, data, time_step):
    historical_predictions = []
    for i in range(time_step, len(data)):
        sequence = data[i-time_step:i]
        pred = model.predict(sequence.reshape(1, time_step, 1))[0, 0]
        historical_predictions.append(pred)
    historical_predictions = np.array(historical_predictions).reshape(-1, 1)
    return historical_predictions

def predict_historical_linear(model, data, time_step):
    X = [data[i:(i + time_step), 0] for i in range(len(data) - time_step)]
    historical_predictions = model.predict(np.array(X))
    historical_predictions = np.array(historical_predictions).reshape(-1, 1)
    return historical_predictions

def predict_future_lstm(model, last_sequence, scaler, forecast_period):
    future_predictions = []
    for _ in range(forecast_period):
        pred = model.predict(last_sequence.reshape(1, len(last_sequence), 1))[0, 0]
        noise = np.random.normal(loc=0, scale=0.025)
        pred = pred * (1 + noise)
        future_predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], pred)
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

def predict_future_linear(model, last_sequence, scaler, forecast_period):
    future_predictions = []
    for _ in range(forecast_period):
        pred = model.predict(last_sequence.reshape(1, -1))[0]
        noise = np.random.normal(loc=0, scale=0.025)
        pred = pred * (1 + noise)
        future_predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], pred)
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

def plot_data(data, historical_predictions, future_predictions, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[30:], y=data['Close'], line={"color": "#00ff37", "width": 1.5}, name="Actual Closing Price"))
    fig.add_trace(go.Scatter(x=data.index[30:], y=historical_predictions[:, 0], line={"color": "#ffc800", "width": 1}, name='Predicted Closing Price'))
    future_dates = pd.date_range(start=data.index[-1], periods=len(future_predictions) + 1, closed='right')
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions[:, 0], mode='lines', line={"color": "#ff4000", "width": 1.5}, name='Future Predicted Price'))
    fig.layout.update(title_text=f"Graph of {symbol}", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def main():
    st.title("Stock Prediction App")
    start_date = "2015-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = ""

    selected_stock = st.text_input("Enter a stock symbol: ", st.session_state.selected_stock)
    random_button = st.button("Get a random symbol")
    if random_button:
        suggest_symbol()
    selected_model = st.selectbox("Select a model: ", [None,"LSTM", "Linear"])
    month_intervals = st.slider("Months of prediction:", 0, 3)
    predict_button = st.button("Predict")

    if predict_button:
        if selected_stock and month_intervals>0 and selected_model != None:
            try:
                if selected_stock in sp500_symbols:
                    data = load_data(selected_stock, start_date, end_date)
                    st.write("<br>", unsafe_allow_html=True)
                    st.success(f'{selected_stock} data successfully loaded!')
                    st.write("<br>", unsafe_allow_html=True)
                    st.subheader(f"{selected_stock} Last 10 Days Data")
                    st.write(data.tail(10))

                    scaled_data, scaler = preprocess_data(data)

                    time_step = 30
                    X, y = create_dataset(scaled_data, time_step)

                    train_size = int(len(data) * 0.8)
                    if train_size >= time_step:
                        X_train, X_test = X[:train_size], X[train_size:]
                        y_train, y_test = y[:train_size], y[train_size:]
                    else:
                        st.error("Not enough data for training. Decrease time_step or fetch more historical data.")
                        return

                    forecast_period = month_intervals * 30
                    if selected_model == "Linear":
                        model = train_linear_model(X_train, y_train)
                        historical_predictions = predict_historical_linear(model, scaled_data, time_step)
                        future_predictions = predict_future_linear(model, scaled_data[-time_step:], scaler, forecast_period)
                    else:
                        model = train_lstm_model(X_train, y_train, epochs=5, batch_size=64)
                        historical_predictions = predict_historical_lstm(model, scaled_data, time_step)
                        future_predictions = predict_future_lstm(model, scaled_data[-time_step:], scaler, forecast_period)

                    historical_predictions = scaler.inverse_transform(historical_predictions)

                    mae, mse, rmse = calculate_metrics(data['Close'].values[time_step:], historical_predictions)

                    plot_data(data, historical_predictions, future_predictions, selected_stock)

                    col1, col2 = st.columns(2)
                    col3 = st.container()

                    with col1:
                        st.subheader("Used Model and Symbol")
                        st.write(f"Symbol: {selected_stock}")
                        st.write(f"Model: {selected_model} Model")
                        st.write(f"Time interval: {month_intervals} Months")

                    with col2:
                        st.subheader("Model Performance Metrics")
                        st.write(f"Mean Absolute Error (MAE): {mae}")
                        st.write(f"Mean Squared Error (MSE): {mse}")
                        st.write(f"Root Mean Squared Error (RMSE): {rmse}")

                    with col3:
                        st.write("<br>", unsafe_allow_html=True)
                        st.subheader("Company Info")

                        col4, col5 = st.columns(2)

                        company_info = get_company_info(selected_stock)

                        with col4:
                            st.write(f"**Name:** {company_info[0]}")
                            st.write(f"**Website:** {company_info[1]}")

                        with col5:
                            st.write(f"**Industry:** {company_info[2]}")
                            st.write(f"**Sector:** {company_info[3]}")

                        st.write(f"**Business Summary:** {company_info[4]} ...")
                else:
                    st.error("You entered an invalid/non-existent symbol. Please try again!")
            except Exception as e:
                st.error(f"An error occurred. Please try again!")
                st.error(f"Error code: {e}")
        else:
            st.error("Please enter a stock symbol, select a model and choose months of prediction above.")

if __name__ == "__main__":
    main()
