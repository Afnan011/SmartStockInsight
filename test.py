import yfinance as yf


def getcurrent_price(stock_ticker):
    stock_data = yf.download(stock_ticker, period="1d")
    return round(stock_data["Close"][0], 2)


print(getcurrent_price('TCS.NS'))