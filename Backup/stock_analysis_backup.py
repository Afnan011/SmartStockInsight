import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



## Loading Stock Data**
# Download stock data for the last 6 months

stock_idx = 1
stock_list = ["TCS", "Tata_Motors", "Infosys", "Asian_Paints", "ONGC"]

ticker = ['TCS.NS', 'TATAMOTORS.NS', 'INFY.NS', 'ASIANPAINT.NS', 'ONGC.NS']

data = yf.download(ticker[stock_idx], period='6mo', interval='1d')

# Use the adjusted closing price
data = data[['Adj Close']]
data.rename(columns={'Adj Close': 'Close'}, inplace=True)

# Calculate simple moving averages and exponential moving averages
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

# Create additional features
data['PRICE_SMA_10'] = data['Close'] - data['SMA_10']
data['PRICE_SMA_20_Ratio'] = data['Close'] / data['SMA_20']
data['EMA_DIFF'] = data['EMA_50'] - data['EMA_200']

data = data.dropna()


# Plot the adjusted closing price and indicators
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Adjusted Closing Price')
plt.plot(data['SMA_10'], label='SMA 10')
plt.plot(data['SMA_20'], label='SMA 20')
plt.plot(data['EMA_50'], label='EMA 50')
plt.plot(data['EMA_200'], label='EMA 200')
plt.title('TCS Adjusted Closing Price and Indicators Last 6 Months')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price INR')
plt.legend()
plt.savefig(r'./images/stock_price_and_indicators.png')



# combine DataFrame
sentiment_data = pd.read_csv(rf'./Dataset/{stock_list[stock_idx]}_sentiment_data.csv')
sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
sentiment_data.set_index('Date', inplace=True)

# Combine with the adjusted closing price data
combined_df = data.join(sentiment_data, how='left')

# Fill any missing sentiment scores (if necessary) - for example, with 0 for neutrality
combined_df.fillna({'Label': 0}, inplace=True)

print(combined_df.head())
print(combined_df.shape)
# combined_df.to_csv('combined_data.csv')




## Preparing the dataset for the model

# Select features and target variable
features = combined_df[['Close','Label','SMA_10', 'SMA_20', 'EMA_50', 'EMA_200']]
target = combined_df['Close']

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 60

X, y = create_dataset(scaled_features, time_step)

# Split the data into training and test sets (80/20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]




## Training the model

# # # Define the model architecture
# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Define early stopping callback
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train the model with early stopping
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
#                     epochs=200, batch_size=32)
# # history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
# #                     epochs=200, batch_size=32, callbacks=[early_stop])



#@ Model 2
# LSTM with GRU

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, X.shape[2])))  # LSTM layer
model.add(Dropout(0.3))  # Dropout layer
model.add(GRU(100, return_sequences=False))  # GRU layer
model.add(Dropout(0.3))  # Dropout layer
model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))  # Dense layer with L2 regularization
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=250, batch_size=64, callbacks=[early_stop])

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print("Training Loss:", train_loss)
print("Testing Loss:", test_loss)

print(model.summary())




# Make predictions on the training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reshape the predictions to fit the scaler's expectations
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

# Invert scaling to get actual values
train_predict_inverse = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], X_train.shape[2]-1))), axis=1))[:, 0]
test_predict_inverse = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], X_test.shape[2]-1))), axis=1))[:, 0]

# Invert scaling for actual values as well
y_train_inverse = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], X_train.shape[2]-1))), axis=1))[:, 0]
y_test_inverse = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2]-1))), axis=1))[:, 0]

# Plot the results
plt.figure(figsize=(13, 6))
plt.plot(combined_df.index[:len(y_train_inverse)], y_train_inverse, label='Actual Train Price')
plt.plot(combined_df.index[:len(train_predict_inverse)], train_predict_inverse, label='Train Predict')
plt.plot(combined_df.index[len(y_train_inverse):len(y_train_inverse) + len(y_test_inverse)], y_test_inverse, label='Actual Test Price')
plt.plot(combined_df.index[len(y_train_inverse):len(y_train_inverse) + len(test_predict_inverse)], test_predict_inverse, label='Test Predict')
plt.title('Stock Price Prediction with Sentiment Analysis')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.savefig(r'./images/stock_price_prediction_training_vs_testing.png')



# Calculate evaluation metrics
train_mae = mean_absolute_error(y_train_inverse, train_predict_inverse)
train_mse = mean_squared_error(y_train_inverse, train_predict_inverse)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train_inverse, train_predict_inverse)

test_mae = mean_absolute_error(y_test_inverse, test_predict_inverse)
test_mse = mean_squared_error(y_test_inverse, test_predict_inverse)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test_inverse, test_predict_inverse)

print(f'Train MAE: {train_mae}')
print(f'Train MSE: {train_mse}')
print(f'Train RMSE: {train_rmse}')
print(f'Train R2: {train_r2}')

print(f'Test MAE: {test_mae}')
print(f'Test MSE: {test_mse}')
print(f'Test RMSE: {test_rmse}')
print(f'Test R2: {test_r2}')




# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]

# Inverse transform the actual values
y_train_actual = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]
y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]

# Calculate metrics for training set
train_mae = mean_absolute_error(y_train_actual, train_predict)
train_mse = mean_squared_error(y_train_actual, train_predict)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train_actual, train_predict)

# Calculate metrics for testing set
if len(y_test_actual) > 1:
    test_mae = mean_absolute_error(y_test_actual, test_predict)
    test_mse = mean_squared_error(y_test_actual, test_predict)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_actual, test_predict)
else:
    test_mae = test_mse = test_rmse = test_r2 = np.nan

print(f'Training Metrics:')
print(f'MAE: {train_mae}')
print(f'MSE: {train_mse}')
print(f'RMSE: {train_rmse}')
print(f'R²: {train_r2}')

print(f'\nTesting Metrics:')
print(f'MAE: {test_mae}')
print(f'MSE: {test_mse}')
print(f'RMSE: {test_rmse}')
print(f'R²: {test_r2}')

# Ensure the indices for plotting match the lengths of the predictions
train_plot_indices = combined_df.index[time_step:time_step + len(train_predict)]
test_plot_indices = combined_df.index[time_step + len(train_predict):time_step + len(train_predict) + len(test_predict)]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(combined_df.index, combined_df['Close'], label='Actual Price')
plt.plot(train_plot_indices, train_predict, label='Train Predict')
plt.plot(test_plot_indices, test_predict, label='Test Predict')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(r'./images/stock_price_prediction_training_vs_testing2.png')



## Predicting

# Predicting the next 30 days
last_days = scaled_features[-time_step:]

num_features = X.shape[2]
predictions = []

current_input = last_days.reshape(1, time_step, num_features)

for i in range(30):
    next_prediction = model.predict(current_input)
    predictions.append(next_prediction[0, 0])
    print(f"step {i+ 1}")

    next_prediction = next_prediction.reshape(1, 1, 1)
    next_prediction_tiled = np.tile(next_prediction, (1, 1, num_features))

    new_input = np.concatenate((current_input[:, 1:, :], next_prediction_tiled), axis=1)
    current_input = new_input

# Invert predictions to get actual values
predictions = np.array(predictions).reshape(-1, 1)
predictions_full = np.concatenate([predictions, np.zeros((predictions.shape[0], num_features - 1))], axis=1)
predictions = scaler.inverse_transform(predictions_full)[:, 0]


print("OUT: Predicted price is: ", predictions.mean())

# Plot the predictions
future_dates = pd.date_range(start=combined_df.index[-1] + pd.Timedelta(days=1), periods=30)
plt.figure(figsize=(12, 6))
plt.plot(combined_df.index, combined_df['Close'], label='Actual Price')
plt.plot(future_dates, predictions, label='Predicted Price')
plt.title('Stock Price Prediction for Next Month')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price INR')
plt.legend()
plt.grid()
plt.savefig(r'./images/stock_price_prediction_next_month.png')



