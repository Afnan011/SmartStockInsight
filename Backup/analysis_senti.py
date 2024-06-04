import random
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
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

def get_random_date():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)

    random_days = random.randint(0, 90)
    random_date = start_date + datetime.timedelta(days=random_days)

    return random_date



stock_idx = 0


# Yahoo finance news

stock_name_yfinance = ["TCS.NS/news", "TATAMOTORS.NS/news", "INFY/latest-news", "ASIANPAINT.NS/news", "ONGC.NS/news"]
url = f"https://finance.yahoo.com/quote/{stock_name_yfinance[stock_idx]}"

headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36' }
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')


news_list = []
get_news = soup.find_all('div', class_='news-stream')

for div in get_news:
    news_items = div.find_all('li', class_='stream-item')

    for item in news_items:
        section = item.find('section')
        if section:
            content_div = section.find('div', class_='content')
            if content_div:
                title_elem = content_div.find('h3')
                description_elem = content_div.find('p')


                if title_elem and description_elem:
                    title = title_elem.get_text().strip()
                    description = description_elem.get_text().strip()
                    news = title+description

                    news_list.append({'Date': get_random_date(), 'News': news})


df_yfinance = pd.DataFrame(news_list)
df_yfinance['Date'] = pd.to_datetime(df_yfinance['Date'])
df_yfinance.set_index('Date', inplace=True)
df_yfinance.sort_values('Date', inplace=True)
# print(df_yfinance)




# NewsAPI 
newsapi = NewsApiClient(api_key='bd55f5254a51444a97868c574afb5726')


# Date range
end_date = datetime.datetime.now().date()
start_date = end_date - datetime.timedelta(days=30)

# Converting the dates to string format
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

stocks_name_news_api = ["Tata Consultancy Services", "Tata Motors",  "Infosys", "Asian Paints", "ONGC"]

all_articles = newsapi.get_everything(q=stocks_name_news_api[stock_idx],
                                      from_param=start_date_str,
                                      to=end_date_str,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=100)


df_news_api = pd.DataFrame(columns=['Date', 'News',])
articles_list = []

for idx, article in enumerate(all_articles['articles'], start=1):
    # source = article['source']['name']
    heading = article['title']
    description = article['description']
    date_time = datetime.datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
    date = date_time.strftime('%Y-%m-%d')
    news = heading + ' ' + description
    articles_list.append({'Date': date,
                          'News': news,
                          })

df_news_api = pd.DataFrame(articles_list)
df_news_api['Date'] = pd.to_datetime(df_news_api['Date'])

df_news_api = df_news_api[df_news_api['News'] != "[Removed] [Removed]"]
df_news_api.set_index('Date', inplace=True)
df_news_api.sort_values('Date', inplace=True)
# print(df_news_api)





## Custom Moneycontrol API
class Api:
    """
    A class used to store constants
    """

    def __init__(self, title_info=None, link_info=None, date_info=None, news_info=None):
        """
        Initializes the constants
        """
        self.Data = {
            "NewsType": news_info,
            "Title": title_info,
            "Link": link_info,
            "Date": date_info,
            "API_CALLED": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.html_parser = "html.parser"
        self.url = [
            "https://www.moneycontrol.com/news",
            "https://www.moneycontrol.com/news/business",
            "https://www.moneycontrol.com/news/latest-news/",
        ]


headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Connection": "keep-alive",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    }

stocks_name_moneycontrol = ["TCS", "TEL", "IT", "API", "ONG"]
MONEY_CONTROL_URL= f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={stocks_name_moneycontrol[stock_idx]}"


soup = BeautifulSoup(requests.get(MONEY_CONTROL_URL, timeout=60, headers=headers).text, Api().html_parser)


news_soup = soup.find_all("div", {"class": "FL"})

news_list = []
date_list = []

for div in news_soup:
    news_title = div.find_all('a', class_='g_14bl')
    date = div.find_all('p', class_='PT3 a_10dgry')

    for data in news_title:
          for element in data:
            if element.name == 'strong':
              news_list.append( element.text.strip())

    for data in date:
      if date:
        text = data.text.strip()
        parts = text.split('|')
        time, date = parts[:2]
        date_list.append(date.strip())


combined_list = []
for i in range(min(len(news_list), len(date_list))):
  combined_list.append({
      "Date": date_list[i].strip() if i < len(date_list) else "NaN" ,
      "News": news_list[i].strip() if i < len(news_list) else ""
    })

df_money_control = pd.DataFrame(combined_list)
df_money_control['Date'] = pd.to_datetime(df_money_control['Date'])

df_money_control.set_index('Date', inplace=True)
df_money_control.sort_values('Date', inplace=True)
# print(df_money_control)




## Google News
stock_name_gnews = ["TCS", "Tata Motors", "Infosys", "Asian Paints", "ONGC"]

googlenews=GoogleNews(start=start_date,end=end_date)
googlenews.search(stock_name_gnews[stock_idx])
result=googlenews.result()
df_gnews=pd.DataFrame(result)

df_gnews = df_gnews.rename(columns={'datetime': 'Date', 'desc': 'News'})

for row in df_gnews.itertuples():
    title = row.title
    desc = row.News
    news = title + ' ' + desc
    df_news_api.loc[row.Index, 'News'] = news

df_gnews['Date'] = df_gnews['Date'].apply(lambda x: datetime.datetime.strptime(datetime.datetime.strftime(x, "%Y-%m-%d"), "%Y-%m-%d").date())
df_gnews['Date'] = pd.to_datetime(df_gnews['Date'])
df_gnews.set_index('Date', inplace=True)
df_gnews.sort_values('Date', inplace=True)


df_gnews = df_gnews.drop(['title', 'media', 'date', 'link', 'img'], axis=1)

# print(df_gnews.shape)
# print(df_gnews)


# combine df[multiple dataframe into 1]

combined_df = pd.concat([df_yfinance, df_news_api, df_gnews, df_money_control])
# print(combined_df)



# Preprocessing
nltk.download('stopwords')

def clean_text(text):
  # Lowercase text
  text = text.lower()

  # Remove punctuation
  text = ''.join([char for char in text if char not in string.punctuation])

  # Remove stop words
  stop_words = stopwords.words('english')
  text = ' '.join([word for word in text.split() if word not in stop_words])

  # Stemming
  stemmer = PorterStemmer()
  text = ' '.join([stemmer.stem(word) for word in text.split()])
  return text



combined_df['News'] = combined_df['News'].apply(clean_text)
# print(combined_df)

combined_df.index= pd.to_datetime(combined_df.index)


combined_df = combined_df[combined_df.index.year >= 2023]
combined_df_sorted = combined_df.sort_index()
# print(combined_df_sorted)



# Sentiment Analysis
classifier = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

for row in combined_df_sorted.itertuples():
  res = classifier(row.News)
  combined_df_sorted.loc[row.Index, 'Score'] = res[0]['score']
  combined_df_sorted.loc[row.Index, 'Label'] = res[0]['label']



for row in combined_df_sorted.itertuples():
  if row.Label == 'positive':
    combined_df_sorted.loc[row.Index, 'Label']  = 1
  elif row.Label == 'neutral':
    combined_df_sorted.loc[row.Index, 'Label']  = 0
  else:
    combined_df_sorted.loc[row.Index, 'Label']  = -1

# print(combined_df_sorted)


# combine news to get new dataset
def aggregate_data(group):
    return pd.Series({
        'Label': group['Label'].max(),
        'Score': group['Score'].mean()
    })

df_grouped = combined_df_sorted.groupby('Date').apply(aggregate_data).reset_index()
# print(df_grouped)




# Plot the count of news by label
sns.countplot(x="Label", data=df_grouped)

# Add title and labels
plt.title("Count of News by Label")
plt.xlabel("Label")
plt.ylabel("Count")

# Save the plot
plt.savefig(r'./images/count_news_by_label.png')




# Classify sentiment based on label sum
df_grouped['Sentiment'] = df_grouped['Label'].apply(lambda x: 'Negative' if x < 0 else ('Neutral' if x == 0 else 'Positive'))

# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df_grouped, palette='viridis', order=['Negative', 'Neutral', 'Positive'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig(r'./images/sentiment_distribution.png')


# Save the grouped DataFrame to a new CSV file
data = df_grouped.copy()

data.set_index('Date', inplace=True)
complete_date_range = pd.date_range(start=data.index.min(), end=data.index.max())

data = data.reindex(complete_date_range)
data.reset_index(inplace=True)
data.rename(columns={'index': 'Date'}, inplace=True)

merged_df = pd.merge(data, df_grouped, on='Date', how='outer', indicator=True)

merged_df.rename(columns={'Label_y': 'Label'}, inplace=True)
merged_df.rename(columns={'Score_y': 'Score'}, inplace=True)
merged_df = merged_df[['Date', 'Label', 'Score']]

# Fill missing 'Score' and 'Label' with neutral values
merged_df['Score'].fillna(0, inplace=True)
merged_df['Label'].fillna(0, inplace=True)


# Display the first few rows of the updated DataFrame
# print(merged_df.head())

# Save the updated DataFrame to a new CSV file
merged_df.to_csv('sentiment_data.csv', index=False)
print(merged_df.head())
print(merged_df.shape())


