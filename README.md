# 🚀 Tesla Stock Price Prediction using LSTM

## 📌 Project Overview
This project focuses on predicting Tesla stock prices using deep learning techniques. A Long Short-Term Memory (LSTM) model is used to analyze historical stock data and forecast future prices.

The project also includes a deployed web application where users can visualize predictions interactively.

---

## 🎯 Objective
- Perform time series analysis on Tesla stock data
- Build a predictive model using LSTM
- Visualize trends and predictions
- Deploy the model using Streamlit

---

## 📊 Dataset
- Dataset used: Tesla stock historical data (TSLA.csv)
- Features:
  - Date
  - Open, High, Low, Close
  - Adjusted Close
  - Volume

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Converted Date column to datetime
- Set Date as index
- Sorted data chronologically
- Selected relevant feature (Adjusted Close price)

### 2. Exploratory Data Analysis (EDA)
- Visualized stock price trends
- Observed long-term growth and fluctuations

### 3. Feature Scaling
- Applied MinMaxScaler to normalize data between 0 and 1

### 4. Sequence Creation
- Created sequences of 60 days to predict next value

### 5. Model Building
- Implemented LSTM model
- Added Dropout layer to prevent overfitting
- Used Dense layer for output

### 6. Model Training
- Trained model on historical data
- Evaluated performance using Mean Squared Error (MSE)

### 7. Prediction
- Predicted future stock prices (next 10 days)

### 8. Deployment
- Built a web app using Streamlit
- Enabled interactive prediction visualization

---

## 📈 Results
- Model successfully captured stock trends
- Predictions closely follow actual values
- Achieved reasonable MSE for time series forecasting

---

## 🌐 Live Demo
👉 https://tesla-stock-prediction-noxtcmwvhoopt2yatxxxgs.streamlit.app/

---

## 🖥️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/tesla-stock-prediction.git
cd tesla-stock-prediction

