import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_stock_data(ticker, period='1y', interval='1d'):
    return yf.download(ticker, period=period, interval=interval)

def calculate_indicators(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['PRICE_SMA_10'] = data['Close'] - data['SMA_10']
    data['PRICE_SMA_20_Ratio'] = data['Close'] / data['SMA_20']
    data['EMA_DIFF'] = data['EMA_50'] - data['EMA_200']

    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    data = data.fillna(method='ffill')
    data = data[(data['Close'] - data['Close'].mean()).abs() < 3 * data['Close'].std()]
    return data.dropna()

def plot_indicators(data, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Adjusted Closing Price')
    plt.plot(data['SMA_10'], label='SMA 10')
    plt.plot(data['SMA_20'], label='SMA 20')
    plt.plot(data['EMA_50'], label='EMA 50')
    plt.plot(data['EMA_200'], label='EMA 200')
    plt.title('Adjusted Closing Price and Indicators Last 6 Months')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def load_sentiment_data(filepath):
    sentiment_data = pd.read_csv(filepath)
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
    sentiment_data.set_index('Date', inplace=True)
    return sentiment_data

def combine_data(price_data, sentiment_data):
    combined_df = price_data.join(sentiment_data, how='left')
    combined_df.fillna({'Label': 0}, inplace=True)
    return combined_df

def scale_features(features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        dataX.append(dataset[i:(i + time_step), :])
        dataY.append(dataset[i + time_step, 0])  # Target is the 'Close' price
    return np.array(dataX), np.array(dataY)

def build_model(input_shape):
    model_input = Input(shape=input_shape)
    lstm = LSTM(100, return_sequences=False)(model_input)
    lstm = Dropout(0.3)(lstm)
    dense = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(lstm)
    output = Dense(1)(dense)
    model = Model(inputs=model_input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(combined_df, y_train_inverse, train_predict_inverse, y_test_inverse, test_predict_inverse, filename):
    plt.figure(figsize=(13, 6))
    plt.plot(combined_df.index[:len(y_train_inverse)], y_train_inverse, label='Actual Train Price')
    plt.plot(combined_df.index[:len(train_predict_inverse)], train_predict_inverse, label='Train Predict')
    plt.plot(combined_df.index[len(y_train_inverse):len(y_train_inverse) + len(y_test_inverse)], y_test_inverse, label='Actual Test Price')
    plt.plot(combined_df.index[len(y_train_inverse):len(y_train_inverse) + len(test_predict_inverse)], test_predict_inverse, label='Test Predict')
    plt.title('Stock Price Prediction with Sentiment Analysis')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_future_predictions(combined_df, future_dates, predictions, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df.index, combined_df['Close'], label='Actual Price')
    plt.plot(future_dates, predictions, label='Predicted Price')
    plt.title('Stock Price Prediction for Next Month')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, combined_df, time_step, output_folder, ticker):
    train_predict = model.predict(X_train).reshape(-1, 1)
    test_predict = model.predict(X_test).reshape(-1, 1)

    def inverse_transform(pred, original_shape):
        full_pred = np.concatenate((pred, np.zeros((pred.shape[0], original_shape - 1))), axis=1)
        return scaler.inverse_transform(full_pred)[:, 0]

    train_predict_inverse = inverse_transform(train_predict, X_train.shape[2])
    test_predict_inverse = inverse_transform(test_predict, X_test.shape[2])
    y_train_inverse = inverse_transform(y_train.reshape(-1, 1), X_train.shape[2])
    y_test_inverse = inverse_transform(y_test.reshape(-1, 1), X_test.shape[2])

    plot_predictions(combined_df, y_train_inverse, train_predict_inverse, y_test_inverse, test_predict_inverse, f"{output_folder}/{ticker}_stock_prediction_training_vs_testing.png")

    return train_predict_inverse, test_predict_inverse, y_train_inverse, y_test_inverse

def calculate_metrics(y_train_inverse, train_predict_inverse, y_test_inverse, test_predict_inverse):
    def metrics(y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }

    train_metrics = metrics(y_train_inverse, train_predict_inverse)
    test_metrics = metrics(y_test_inverse, test_predict_inverse) if len(y_test_inverse) > 1 else {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    return {'train': train_metrics, 'test': test_metrics}

def predict_future(model, last_days, time_step, num_features, scaler):
    predictions = []
    current_input = last_days.reshape(1, time_step, num_features)

    for _ in range(30):
        next_prediction = model.predict(current_input)[0, 0]
        predictions.append(next_prediction)
        next_prediction_tiled = np.tile(next_prediction, (1, 1, num_features))
        current_input = np.concatenate((current_input[:, 1:, :], next_prediction_tiled), axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_full = np.concatenate([predictions, np.zeros((predictions.shape[0], num_features - 1))], axis=1)
    return scaler.inverse_transform(predictions_full)[:, 0]

def main(stock_idx=0, epochs_count=100):
    stock_list = ["TCS", "Tata_Motors", "Infosys", "Asian_Paints", "Tech_Mahindra_Ltd"]
    ticker = ['TCS.NS', 'TATAMOTORS.NS', 'INFY.NS', 'ASIANPAINT.NS', 'TECHM.NS']
    stock_ticker = ticker[stock_idx]

    stock_data = load_stock_data(stock_ticker)
    stock_data = calculate_indicators(stock_data)
    plot_indicators(stock_data, f'./images/{stock_list[stock_idx]}_indicators.png')

    sentiment_data = load_sentiment_data(rf'./Dataset/{stock_list[stock_idx]}_sentiment_data.csv')
    combined_df = combine_data(stock_data, sentiment_data)

    # Ensure 'Close' is the first column
    combined_df = combined_df[['Close'] + [col for col in combined_df.columns if col != 'Close']]

    features = combined_df.drop(columns=['Close'])
    target = combined_df[['Close']]

    print("Features: ", features)

    scaled_features, features_scaler = scale_features(features)
    scaled_target, target_scaler = scale_features(target)

    time_step = 30
    X, y = create_dataset(np.hstack((scaled_target, scaled_features)), time_step)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_model((time_step, X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs_count, batch_size=64, callbacks=[early_stop])

    train_predict_inverse, test_predict_inverse, y_train_inverse, y_test_inverse = evaluate_model(model, X_train, y_train, X_test, y_test, target_scaler, combined_df, time_step, './images', stock_list[stock_idx])
    
    metrics = calculate_metrics(y_train_inverse, train_predict_inverse, y_test_inverse, test_predict_inverse)
    print(f"Metrics: {metrics}")

    last_days = X_test[-1]
    future_predictions = predict_future(model, last_days, time_step, X.shape[2], target_scaler)
    print(future_predictions)
    print(future_predictions.mean())


    future_dates = pd.date_range(start=combined_df.index[-1], periods=30, freq='B')
    plot_future_predictions(combined_df, future_dates, future_predictions, f"./static/images/{stock_list[stock_idx]}_stock_price_prediction_next_month.png")

if __name__ == "__main__":
    main(stock_idx=0, epochs_count=100)
