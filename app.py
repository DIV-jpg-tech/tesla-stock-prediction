import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Tesla Stock Price Prediction")
st.write("Predicting next 10 days stock price (Lightweight Model)")

# Load dataset
data = pd.read_csv('TSLA.csv')

# Preprocess
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use only Adj Close
data = data[['Adj Close']]
data.rename(columns={'Adj Close': 'Price'}, inplace=True)

# Show dataset preview
st.subheader("Dataset Preview")
st.write(data.tail())

# Button
if st.button("Predict Future Prices"):

    # Take last 60 days prices
    last_60 = data['Price'][-60:].values
    temp = list(last_60)

    future_predictions = []

    # Predict next 10 days using moving average
    for i in range(10):
        pred = np.mean(temp[-60:])
        future_predictions.append(pred)
        temp.append(pred)

    future_predictions = np.array(future_predictions)

    # Plot
    st.subheader("Next 10 Days Prediction")

    fig, ax = plt.subplots()

    # Last 100 days actual data
    last_100 = data['Price'][-100:].values
    ax.plot(last_100, label='Last 100 Days')

    # Future predictions
    future_range = np.arange(100, 110)
    ax.plot(future_range, future_predictions, label='Next 10 Days', color='red')

    ax.legend()
    st.pyplot(fig)
