import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_stock_data(ticker, period='6mo', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data = data[['Adj Close']]
    data.rename(columns={'Adj Close': 'Close'}, inplace=True)
    return data

def calculate_indicators(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['PRICE_SMA_10'] = data['Close'] - data['SMA_10']
    data['PRICE_SMA_20_Ratio'] = data['Close'] / data['SMA_20']
    data['EMA_DIFF'] = data['EMA_50'] - data['EMA_200']
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
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def build_model(input_shape):
    # LSTM model
    lstm_input = Input(shape=input_shape)
    lstm = Bidirectional(LSTM(100, return_sequences=True))(lstm_input)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(LSTM(100, return_sequences=False))(lstm)
    lstm = Dropout(0.3)(lstm)
    lstm_output = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(lstm)
    
    # GRU model
    gru_input = Input(shape=input_shape)
    gru = Bidirectional(GRU(100, return_sequences=True))(gru_input)
    gru = Dropout(0.3)(gru)
    gru = Bidirectional(GRU(100, return_sequences=False))(gru)
    gru = Dropout(0.3)(gru)
    gru_output = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(gru)

    # CNN model
    cnn_input = Input(shape=input_shape)
    cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(cnn_input)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    cnn_output = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(cnn)

    # Concatenate LSTM, GRU, and CNN outputs
    combined = concatenate([lstm_output, gru_output, cnn_output])
    combined_output = Dense(1)(combined)

    model = Model(inputs=[lstm_input, gru_input, cnn_input], outputs=combined_output)
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

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, combined_df, time_step, output_folder):
    train_predict = model.predict([X_train, X_train, X_train])
    test_predict = model.predict([X_test, X_test, X_test])
    
    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)
    
    train_predict_inverse = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], X_train.shape[2]-1))), axis=1))[:, 0]
    test_predict_inverse = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], X_test.shape[2]-1))), axis=1))[:, 0]
    
    y_train_inverse = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], X_train.shape[2]-1))), axis=1))[:, 0]
    y_test_inverse = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2]-1))), axis=1))[:, 0]
    
    plot_predictions(combined_df, y_train_inverse, train_predict_inverse, y_test_inverse, test_predict_inverse, f"{output_folder}/stock_price_prediction_training_vs_testing.png")

    return train_predict_inverse, test_predict_inverse, y_train_inverse, y_test_inverse

def calculate_metrics(y_train_inverse, train_predict_inverse, y_test_inverse, test_predict_inverse):
    train_mae = mean_absolute_error(y_train_inverse, train_predict_inverse)
    train_mse = mean_squared_error(y_train_inverse, train_predict_inverse)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_inverse, train_predict_inverse)

    if len(y_test_inverse) > 1:
        test_mae = mean_absolute_error(y_test_inverse, test_predict_inverse)
        test_mse = mean_squared_error(y_test_inverse, test_predict_inverse)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test_inverse, test_predict_inverse)
    else:
        test_mae = test_mse = test_rmse = test_r2 = np.nan

    metrics = {
        'train': {
            'MAE': train_mae,
            'MSE': train_mse,
            'RMSE': train_rmse,
            'R2': train_r2,
        },
        'test': {
            'MAE': test_mae,
            'MSE': test_mse,
            'RMSE': test_rmse,
            'R2': test_r2,
        }
    }
    
    return metrics

def predict_future(model, last_days, time_step, num_features, scaler):
    predictions = []
    current_input = last_days.reshape(1, time_step, num_features)

    for i in range(30):
        next_prediction = model.predict(current_input)
        predictions.append(next_prediction[0, 0])
        next_prediction = next_prediction.reshape(1, 1, 1)
        next_prediction_tiled = np.tile(next_prediction, (1, 1, num_features))
        new_input = np.concatenate((current_input[:, 1:, :], next_prediction_tiled), axis=1)
        current_input = new_input

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_full = np.concatenate([predictions, np.zeros((predictions.shape[0], num_features - 1))], axis=1)
    predictions = scaler.inverse_transform(predictions_full)[:, 0]
    
    return predictions

def main():
    stock_idx = 0
    stock_list = ["TCS", "Tata_Motors", "Infosys", "Asian_Paints", "ONGC"]
    ticker = ['TCS.NS', 'TATAMOTORS.NS', 'INFY.NS', 'ASIANPAINT.NS', 'ONGC.NS']
    stock_ticker = ticker[stock_idx]

    stock_data = load_stock_data(stock_ticker)
    stock_data = calculate_indicators(stock_data)
    plot_indicators(stock_data, './images/indicators.png')

    sentiment_data = pd.read_csv(rf'./Dataset/{stock_list[stock_idx]}_sentiment_data.csv')
    combined_df = combine_data(stock_data, sentiment_data)

    features = combined_df[['Close', 'PRICE_SMA_10', 'PRICE_SMA_20_Ratio', 'EMA_DIFF', 'Label']]
    scaled_features, scaler = scale_features(features)
    time_step = 60
    X, y = create_dataset(scaled_features, time_step)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    input_shape = (time_step, X.shape[2])
    model = build_model(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit([X_train, X_train, X_train], y_train, validation_data=([X_test, X_test, X_test], y_test), epochs=250, batch_size=64, callbacks=[early_stop])

    train_predict_inverse, test_predict_inverse, y_train_inverse, y_test_inverse = evaluate_model(model, X_train, y_train, X_test, y_test, scaler, combined_df, time_step, './images')

    metrics = calculate_metrics(y_train_inverse, train_predict_inverse, y_test_inverse, test_predict_inverse)
    for dataset, metric in metrics.items():
        print(f"{dataset.capitalize()} Metrics:")
        for metric_name, value in metric.items():
            print(f"{metric_name}: {value}")
        print()

    last_days = scaled_features[-time_step:]
    predictions = predict_future(model, last_days, time_step, X.shape[2], scaler)
    future_dates = pd.date_range(start=combined_df.index[-1] + pd.Timedelta(days=1), periods=30)
    plot_future_predictions(combined_df, future_dates, predictions, './images/stock_price_prediction_next_month.png')

if __name__ == "__main__":
    main()
