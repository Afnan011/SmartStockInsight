import random
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import nltk
from newsapi import NewsApiClient
from GoogleNews import GoogleNews
from newspaper import Article
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize NLTK resources
nltk.download('stopwords')

def get_random_date():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    random_days = random.randint(0, 90)
    return start_date + datetime.timedelta(days=random_days)

def fetch_yahoo_finance_news(stock_idx):
    stock_name_yfinance = ["TCS.NS/news", "TATAMOTORS.NS/news", "INFY/latest-news", "ASIANPAINT.NS/news", "TECHM.NS/news/"]
    url = f"https://finance.yahoo.com/quote/{stock_name_yfinance[stock_idx]}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Error fetching Yahoo Finance news: {e}")
        return pd.DataFrame()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    news_list = []
    
    for div in soup.find_all('div', class_='news-stream'):
        for item in div.find_all('li', class_='stream-item'):
            section = item.find('section')
            if section:
                content_div = section.find('div', class_='content')
                if content_div:
                    title_elem = content_div.find('h3')
                    description_elem = content_div.find('p')
                    if title_elem and description_elem:
                        title = title_elem.get_text().strip()
                        description = description_elem.get_text().strip()
                        news_list.append({'Date': get_random_date(), 'News': title + description})
    
    df = pd.DataFrame(news_list)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_values('Date', inplace=True)
    return df

def fetch_newsapi_news(stock_idx):
    api_key = os.environ['NewsAPI_TOKEN']
    newsapi = NewsApiClient(api_key=api_key)
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=30)
    
    stocks_name_news_api = ["Tata Consultancy Services", "Tata Motors", "Infosys", "Asian Paints", "Tech Mahindra Ltd"]
    all_articles = newsapi.get_everything(q=stocks_name_news_api[stock_idx],
                                          from_param=start_date.strftime('%Y-%m-%d'),
                                          to=end_date.strftime('%Y-%m-%d'),
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100)
    articles_list = []
    
    for article in all_articles['articles']:
        heading = article['title'] if article['title'] is not None else ''
        description = article['description'] if article['description'] is not None else ''
        date = article['publishedAt'].split('T')[0]
        news = heading + ' ' + description
        articles_list.append({'Date': date, 'News': news})
    
    df = pd.DataFrame(articles_list)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_values('Date', inplace=True)
    return df

def fetch_moneycontrol_news(stock_idx):
    stocks_name_moneycontrol = ["TCS", "TEL", "IT", "API", "TM4"]
    url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={stocks_name_moneycontrol[stock_idx]}"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Error fetching Moneycontrol news: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')
    news_list = []
    date_list = []

    for div in soup.find_all("div", {"class": "FL"}):
        news_title = div.find_all('a', class_='g_14bl')
        date = div.find_all('p', class_='PT3 a_10dgry')

        for data in news_title:
            for element in data:
                if element.name == 'strong':
                    news_list.append(element.text.strip())

        for data in date:
            if data:
                text = data.text.strip()
                parts = text.split('|')
                date_list.append(parts[1].strip())

    combined_list = [{'Date': date_list[i], 'News': news_list[i]} for i in range(min(len(news_list), len(date_list)))]

    df = pd.DataFrame(combined_list)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_values('Date', inplace=True)
    return df

def fetch_google_news(stock_idx):
    stock_name_gnews = ["TCS", "Tata Motors", "Infosys", "Asian Paints", "Tech Mahindra Ltd"]
    googlenews = GoogleNews()
    googlenews.search(stock_name_gnews[stock_idx])
    result = googlenews.result()
    df = pd.DataFrame(result)
    
    if not df.empty:
        df = df.rename(columns={'datetime': 'Date', 'desc': 'News'})
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_values('Date', inplace=True)
        df.drop(['title', 'media', 'date', 'link', 'img'], axis=1, inplace=True)
    return df

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def perform_sentiment_analysis(df):
    classifier = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    for row in df.itertuples():
        res = classifier(row.News)
        df.at[row.Index, 'Score'] = res[0]['score']
        df.at[row.Index, 'Label'] = res[0]['label']

    df['Label'] = df['Label'].map({'positive': 1, 'neutral': 0, 'negative': -1})
    return df

def aggregate_data(group):
    return pd.Series({
        'Label': group['Label'].max(),
        'Score': group['Score'].mean()
    })



def plot_news_count(df, stock_ticker):
    sns.countplot(x="Label", data=df)
    plt.title("Count of News by Label")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(rf'./images/{stock_ticker}_count_news_by_label.png')

def plot_sentiment_distribution(df, stock_ticker):
    df['Sentiment'] = df['Label'].apply(lambda x: 'Negative' if x < 0 else ('Neutral' if x == 0 else 'Positive'))
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment', data=df, palette='viridis', order=['Negative', 'Neutral', 'Positive'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(rf'./images/{stock_ticker}_sentiment_distribution.png')

# def plot_sentiment_pie_chart(df, stock_ticker):
#     sentiment_counts = df['Label'].value_counts()
#     sentiment_labels = ['Positive' if x == 1 else 'Neutral' if x == 0 else 'Negative' for x in sentiment_counts.index]
#     colors = ['green', 'grey', 'red']
    
#     plt.figure(figsize=(8, 6))
#     plt.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', colors=colors, startangle=140)
#     plt.title(f'Sentiment Distribution for {stock_ticker}')
    
#     output_dir = f'./static/images/'
#     plt.savefig(os.path.join(output_dir, f'{stock_ticker}_sentiment_graph.png'))

def plot_sentiment_pie_chart(df, stock_ticker):
    sentiment_counts = df['Label'].value_counts().sort_index()
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    colors = ['red', 'gray', 'green']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title(f'Sentiment Distribution for {stock_ticker}')
    
    output_dir = './static/images/'
    plt.savefig(os.path.join(output_dir, f'{stock_ticker}_sentiment_graph.png'))

def main(stock_idx=0):
    stock_list = ["TCS", "Tata_Motors", "Infosys", "Asian_Paints", "Tech_Mahindra_Ltd"]
    logging.info(f"Fetching news for {stock_list[stock_idx]}")
    df_yfinance = fetch_yahoo_finance_news(stock_idx)
    df_news_api = fetch_newsapi_news(stock_idx)
    df_money_control = fetch_moneycontrol_news(stock_idx)
    df_gnews = fetch_google_news(stock_idx)

    combined_df = pd.concat([df_yfinance, df_news_api, df_gnews, df_money_control])
    combined_df['News'] = combined_df['News'].apply(clean_text)
    combined_df = combined_df[combined_df.index.year >= 2023]
    combined_df = combined_df.sort_index()

    combined_df = perform_sentiment_analysis(combined_df)
    df_grouped = combined_df.groupby('Date').apply(aggregate_data).reset_index()

    # plot_news_count(df_grouped, stock_list[stock_idx])
    plot_sentiment_distribution(df_grouped, stock_list[stock_idx])
    plot_sentiment_pie_chart(df_grouped, stock_list[stock_idx])


    df_grouped.set_index('Date', inplace=True)
    
    complete_date_range = pd.date_range(start=df_grouped.index.min(), end=df_grouped.index.max())
    df_grouped = df_grouped.reindex(complete_date_range).reset_index().rename(columns={'index': 'Date'})
    df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])

    df_grouped['Label'].fillna(0, inplace=True)

    df_grouped.drop(['Sentiment', 'Score'], axis=1, inplace=True)

    filename = f'{stock_list[stock_idx]}_sentiment_data.csv'
    df_grouped.to_csv(rf'./Dataset/{filename}', index=False)

    logging.info(df_grouped.head())
    logging.info(df_grouped.shape)


if __name__ == "__main__":
    stock_list = ["TCS", "Tata_Motors", "Infosys", "Asian_Paints", "Tech_Mahindra_Ltd"]

    # for idx in range(len(stock_list)):
    #     main(idx)


    # main(0)
    main(4)