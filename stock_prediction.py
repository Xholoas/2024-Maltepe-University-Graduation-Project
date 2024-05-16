import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from datetime import date
import streamlit as st
import yfinance as yf

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go

def load_custom_css():
    with open("custom.css","r") as file:
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

def forecast_lstm(model, data, scaler, time_step, forecast_period):
    last_sequence = data[-time_step:]
    forecast = []
    for x in range(forecast_period):
        current_pred = model.predict(last_sequence.reshape(1, time_step, 1))[0, 0]
        noise = np.random.normal(loc=0, scale=0.015)
        current_pred_with_noise = current_pred * (1 + noise)
        forecast.append(current_pred_with_noise)
        last_sequence = np.append(last_sequence[1:], current_pred_with_noise)
    forecast = np.array(forecast).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast)
    return forecast

def forecast_linear(model, data, scaler, time_step, forecast_period):
    last_sequence = data[-time_step:]
    forecast = []
    for x in range(forecast_period):
        current_pred = model.predict(last_sequence.reshape(1, -1))[0]
        noise = np.random.normal(loc=0, scale=0.015)
        current_pred_with_noise = current_pred * (1 + noise)
        forecast.append(current_pred_with_noise)
        last_sequence = np.append(last_sequence[1:], current_pred_with_noise)
    forecast = np.array(forecast).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast)
    return forecast

def plot_data(data, forecast, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], line={"color": "#508D69","width":1.5}, name="Closing Price"))
    fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=len(forecast)+1)[1:], y=forecast[:, 0], mode='lines', line={"color": "#f5b642","width":1.5}, name='Forecasted Price'))
    fig.layout.update(title_text=f"Graph of {symbol}", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def main():
    st.title("Stock Prediction App")
    st.markdown(f"<style>{load_custom_css()}</style>", unsafe_allow_html=True)
    start_date = "2015-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    selected_stock = st.text_input("Enter a stock symbol: ", "")
    selected_model = st.selectbox("Select a model: ", ["LSTM", "Linear"])
    month_intervals = st.slider("Months of prediction:", 1, 4)
    predict_button = st.button("Predict")

    if predict_button:
        if selected_stock and month_intervals and selected_model:
            try:
                tickers = yf.Tickers(selected_stock).tickers
                if tickers[selected_stock] and not any(char.isdigit() for char in selected_stock):
                    data = load_data(selected_stock, start_date, end_date)
                    st.success(f'{selected_stock} data successfully loaded!')
                    st.subheader(f"{selected_stock} Raw Data")
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

                    if selected_model == "Linear":
                        model = train_linear_model(X_train, y_train)
                        forecast = forecast_linear(model, scaled_data, scaler, time_step, month_intervals * 30)
                    else:
                        model = train_lstm_model(X_train, y_train, epochs=5, batch_size=64)
                        forecast = forecast_lstm(model, scaled_data, scaler, time_step, month_intervals * 30)

                    plot_data(data, forecast, selected_stock)
                else:
                    st.error("You entered an invalid/non-existent symbol. Please try again!")
            except Exception as e:
                st.error("An error occurred. Please try again!")
                print(e)
        else:
            st.error("Please enter a stock symbol, select a model, and choose months of prediction above.")


if __name__ == "__main__":
    main()
