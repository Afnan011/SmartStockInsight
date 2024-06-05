from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import yfinance as yf
import textwrap

app = Flask(__name__)

def get_stock_details(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    stock_info = stock.info
    stock_data = yf.download(stock_ticker, period="1d")
    current_price = round(stock_data["Close"][0], 2)

    long_description = stock_info.get('longBusinessSummary', 'N/A')
    
    wrapper = textwrap.TextWrapper(width=150, max_lines=4, placeholder="...")
    short_description = "\n".join(wrapper.wrap(long_description))

    return {
        'name': stock_info.get('shortName', 'N/A'),
        'sector': stock_info.get('sector', 'N/A'),
        'industry': stock_info.get('industry', 'N/A'),
        'website': stock_info.get('website', 'N/A'),
        'description': short_description,
        'current_price': current_price,
        'fundamental_analysis': get_fundamental_analysis(stock_info)
    }

def get_fundamental_analysis(stock_info):
    return {
        'market_cap': stock_info.get('marketCap', 'N/A'),
        'pe_ratio': stock_info.get('trailingPE', 'N/A'),
        'dividend_yield': stock_info.get('dividendYield', 'N/A')
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    stock_list = ["TCS", "Tata_Motors", "Infosys", "Asian_Paints", "ONGC"]
    ticker_list = ['TCS.NS', 'TATAMOTORS.NS', 'INFY.NS', 'ASIANPAINT.NS', 'ONGC.NS']
    stock_dict = dict(zip(stock_list, ticker_list))

    selected_stock = None
    predicted_price = None
    prediction_graph = None
    sentiment_graph = None
    stock_details = None
    recommendation = None
    fundamental_insights= None

    if request.method == 'POST':
        selected_stock = request.form['stock']
        stock_ticker = stock_dict[selected_stock]
        stock_details = get_stock_details(stock_ticker)
        predicted_price = get_predicted_price(selected_stock)
        prediction_graph = f"{selected_stock}_stock_price_prediction_next_month.png"
        sentiment_graph = f'{selected_stock}_sentiment_graph.png'

        if stock_details['current_price'] != 'N/A' and predicted_price is not None:
            current_price = stock_details['current_price']
            predicted_price = float(predicted_price)
            if predicted_price > current_price:
                recommendation = 'Buy'
            elif predicted_price < current_price:
                recommendation = 'Sell'
            else:
                recommendation = 'Hold'
                
        fundamental_insights = get_fundamental_insights(
            stock_details['fundamental_analysis']['market_cap'],
            stock_details['fundamental_analysis']['pe_ratio'],
            stock_details['fundamental_analysis']['dividend_yield']
        )
        print(fundamental_insights)

    return render_template('index.html', stock_list=stock_list, selected_stock=selected_stock, 
                           predicted_price=predicted_price, prediction_graph=prediction_graph, sentiment_graph=sentiment_graph,
                           stock_details=stock_details, recommendation=recommendation, fundamental_insights=fundamental_insights)

def get_predicted_price(stock_name):
    file_path = './predictions/stock_price_prediction.csv'
    if os.path.exists(file_path):
        predictions_df = pd.read_csv(file_path, index_col=0)
        if stock_name in predictions_df['Stock'].values:
            return predictions_df.loc[predictions_df['Stock'] == stock_name, 'Prediction'].values[0]
    return None

def get_fundamental_insights(market_cap, pe_ratio, dividend_yield):
    insights = []

    if market_cap > 1000000000000:
        insights.append("The company has a large market capitalization, indicating its significant presence in the market.")
    else:
        insights.append("The company has a relatively smaller market capitalization.")

    if pe_ratio < 15:
        insights.append("The PE ratio is low, suggesting the stock might be undervalued.")
    elif pe_ratio > 25:
        insights.append("The PE ratio is high, indicating the stock might be overvalued.")
    else:
        insights.append("The PE ratio is within a reasonable range.")

    if dividend_yield > 0.03:
        insights.append("The company offers a relatively high dividend yield, making it attractive for income investors.")
    else:
        insights.append("The dividend yield is relatively low.")

    return " ".join(insights)



@app.route('/stock_chart_data')
def stock_chart_data():
    stock_ticker = request.args.get('stock')
    if not stock_ticker:
        return jsonify([])

    stock = yf.Ticker(stock_ticker)
    hist = stock.history(period="1y")
    data = [
        {
            "x": date.isoformat(),
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "close": row["Close"]
        } for date, row in hist.iterrows()
    ]
    return jsonify(data)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    app.run(debug=True)
    
