import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("Tesla Stock Price Prediction")
st.write("Predicting next 10 days stock price using LSTM")

# Load dataset
data = pd.read_csv('data/TSLA.csv')

# Preprocess
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use only Adj Close
data = data[['Adj Close']]
data.rename(columns={'Adj Close': 'Price'}, inplace=True)

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, time_steps=60):
    X = []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
    return np.array(X)

# Load last 60 days
last_60_days = scaled_data[-60:]

# Button
if st.button("Predict Future Prices"):
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(60,1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Dummy training (quick)
    X = create_sequences(scaled_data)
    y = scaled_data[60:]
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
    
    temp_input = last_60_days.tolist()
    future_predictions = []

    for i in range(10):
        x_input = np.array(temp_input[-60:])
        x_input = x_input.reshape(1, 60, 1)
        
        pred = model.predict(x_input, verbose=0)
        future_predictions.append(pred[0][0])
        temp_input.append(pred[0])

    # Convert back
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1,1)
    )

    # Plot
    st.subheader("Next 10 Days Prediction")

    fig, ax = plt.subplots()
    last_100 = data['Price'][-100:].values
    ax.plot(last_100, label='Last 100 Days')
    
    future_range = np.arange(100, 110)
    ax.plot(future_range, future_predictions, label='Next 10 Days', color='red')
    
    ax.legend()
    st.pyplot(fig)