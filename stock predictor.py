import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Fetch Data
ticker = 'DJI'  # Change this to any other ticker like 'TSLA', 'INFY', etc.
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
data = data[['Close']]  # Using only the closing price

# Step 2: Create Features
data['Prediction'] = data[['Close']].shift(-30)  # Predict 30 days into the future

# Step 3: Prepare data for training
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Test prediction
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

# Step 7: Predict future
future = data.drop(['Prediction'], axis=1)[-30:]
future_prediction = model.predict(future)

# Step 8: Plotting
plt.figure(figsize=(10,6))
plt.plot(data.index[-60:-30], data['Close'][-60:-30], label='Actual Prices')
plt.plot(data.index[-30:], future_prediction, label='Predicted Prices')
plt.legend()
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()
