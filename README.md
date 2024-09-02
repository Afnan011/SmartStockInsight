# Stock Prediction Web App

This project is a web application for predicting stock prices using sentiment analysis and historical stock data. The application allows users to select a stock, view its fundamental analysis, see the predicted stock price, get recommendations, and visualize live stock price charts.

## Features

- **Stock Selection**: Choose from a list of stocks to analyze.
- **Sentiment Analysis**: Integrate sentiment analysis results from news articles.
- **Historical Data**: Fetch historical stock prices using Yahoo Finance.
- **Stock Details**: Display detailed information about the selected stock.
- **Prediction**: Show predicted stock prices based on historical data and sentiment analysis.
- **Recommendation**: Provide buy, sell, or hold recommendations.
- **Live Charts**: Interactive charts for live stock price data with options for line or candlestick views.

## Technologies Used

- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Data Analysis**: pandas, numpy, scikit-learn, yfinance
- **Machine Learning Models**: TensorFlow, Keras
- **Deployment**: Docker

<!-- ## Machine Learning Models -->

<!-- The prediction model is built using a combination of Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Convolutional Neural Network (CNN) layers. This combination allows the model to capture both temporal dependencies and spatial features in the stock price data, enhancing the prediction accuracy. -->

## Installation

### Prerequisites

- Docker
- Python 3.11

### Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/stock-prediction-web-app.git
    cd stock-prediction-web-app
    ```

2. **Create `requirements-docker.txt`**:
    ```sh
    pip freeze | Select-String -NotMatch "pywin32|tensorflow-intel" > requirements-docker.txt
    ```

3. **Build the Docker image**:
    ```sh
    docker build -t stock-prediction-app .
    ```

4. **Run the Docker container**:
    ```sh
    docker run -p 5000:5000 --name stock-prediction-container stock-prediction-app
    ```

5. **Access the application**:
    Open your web browser and navigate to `http://localhost:5000`.

## File Structure

- `app.py`: The main Flask application file.
- `sentiment.py`: Script for generating sentiment analysis data.
- `stock_analysis.py`: Script for analyzing and predicting stock prices.
- `requirements.txt`: Python dependencies for local development.
- `requirements-docker.txt`: Python dependencies for Docker deployment.
- `Dockerfile`: Docker configuration file.
- `templates/index.html`: HTML template for the web application.
- `static/`: Folder containing static files like CSS.
- `images/`: Folder for storing generated prediction graphs.
- `predictions/`: Folder for storing CSV files with predicted prices.

## Usage

1. **Select Stock**: Choose a stock from the listed stock.
2. **View Details**: See stock details, current price, predicted price, and analysis.
3. **Analyze Graphs**: View the prediction graph and live stock price chart.

## Preview

You can preview the deployed website [here](https://smartstockinsight.onrender.com/). 


## Report
You can view entire project report or document from [here](./Report/4VP22MC027_MAHAMMAD_AFNAN_M.pdf).

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

- **Mahammad Afnan M**

For any inquiries or support, please contact [muhammedafnan8184@gmail.com](mailto://muhammedafnan8184@gmail.com).
