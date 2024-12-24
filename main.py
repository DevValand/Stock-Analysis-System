import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import messagebox

# Fetching stock data
def fetch_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Display historical data
def display_data(data):
    print(data.head())
    print(data.describe())

# Calculate moving averages
def add_moving_averages(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

# Visualize the data with multiple types of graphs
def plot_data(data, ticker):
    plt.figure(figsize=(18, 10))

    # Plot 1: Open Price
    plt.subplot(2, 2, 1)
    plt.plot(data['Open'], label="Open Price", color="blue")
    plt.xlabel("Years")
    plt.ylabel("Open Prices")
    plt.title(f"{ticker} - Historical Open Price")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust space between plots

    plt.grid()
    plt.legend()

    # Plot 2: Close Price Only
    plt.subplot(2, 2, 2)
    plt.plot(data['Close'], label="Close Price", color="green")
    plt.xlabel("Years")
    plt.ylabel("Close Prices")
    plt.title(f"{ticker} - Historical Close Price")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust space between plots    
    plt.grid()
    plt.legend()

    # Plot 3: Moving Averages Only
    plt.subplot(2, 2, 3)
    plt.plot(data['MA20'], label="20-Day MA", color="red")
    plt.plot(data['MA50'], label="50-Day MA", color="green")
    plt.plot(data['MA200'], label="200-Day MA", color="purple")
    plt.xlabel("Years")
    plt.ylabel("Moving Averages")
    plt.title(f"{ticker} - Moving Averages")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust space between plots
    plt.grid()
    plt.legend()

    # Plot 4: Predictions vs Actual Prices (for Linear Regression)
    plt.subplot(2, 2, 4)
    plt.scatter(X_test, y_test, color='blue', label="Actual Price", s=3)
    plt.plot(X_test, y_pred, color='red', label="Predicted Price")
    plt.xlabel("Days since start")
    plt.ylabel("Close Price")
    plt.title("Stock Price Prediction using Linear Regression")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust space between plots
    plt.grid()
    plt.legend(loc="upper left")

      # Plot 5: Candlestick + Volume Chart
    plt.plot(1,1)
    mpf.plot(data[-60:], type='candle', style='charles', volume=True, title=f"{ticker} - Last 60 Days Candlestick + Volume Chart")

    plt.tight_layout()
    plt.show()

# Prepare data for linear regression
def prepare_data(data):
    data = data[['Close']].dropna()
    data['Years'] = (data.index - data.index.min()).days  # Converting dates to numerical values
    X = data['Years'].values.reshape(-1, 1)
    y = data['Close'].values
    return X, y

# Train the linear regression model and predict
def predict_stock_price(X, y):
    global X_test, y_test, y_pred  # For use in the plot_data function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    return model

# Predict future price
def predict_future_price(model, days_ahead):
    future_day = np.array([[days_ahead]])
    future_price = model.predict(future_day)
    return future_price[0]

# GUI Function
def on_submit():
    ticker = entry.get().upper()
    if not ticker:
        messagebox.showerror("Error", "Please enter a valid stock ticker symbol!")
        return

    data = fetch_stock_data(ticker)

    if data.empty:
        messagebox.showerror("Error", "No data found for the given ticker symbol.")
        return

    display_data(data)
    add_moving_averages(data)
    global X, y  # For use in the plot_data function
    X, y = prepare_data(data)
    model = predict_stock_price(X, y)


    # Predict price 30 days after the latest data
    latest_day = (data.index[-1] - data.index[0]).days
    predicted_price = predict_future_price(model, latest_day + 30)
    messagebox.showinfo("Prediction", f"Predicted price for {ticker} 30 days ahead: $/\u20B9 {predicted_price:.2f}")
    print("Prediction", f"Predicted price for {ticker} 30 days ahead: $/\u20B9 {predicted_price:.2f}")

    plot_data(data, ticker)



# Main function with GUI
def main():
    global entry

    root = tk.Tk()
    root.title("Stock Price Predictor")

    # Create input field and button
    tk.Label(root, text="Enter Stock Ticker (e.g., AAPL,TSLA,ADANIENT.NS and RELIANCE.NS):", font=("Arial", 14)).pack(pady=10)
    entry = tk.Entry(root, font=("Arial", 14), width=20)
    entry.pack(pady=5)

    tk.Button(root, text="Submit", command=on_submit, font=("Arial", 14)).pack(pady=20)

    root.mainloop()

# Run the main function
if __name__ == "__main__":
    main()