import requests
import pandas as pd
from datetime import datetime
import os
import time

os.makedirs("data/raw/news", exist_ok=True)

API_KEY = "1d3f1909bd7e4c6a9e61098bf21e5d4a"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "XOM", "UNH"
]

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

all_news = []

for ticker in TICKERS:
    print(f"Fetching {ticker}...")

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}"
        f"&from={START_DATE}&to={END_DATE}"
        f"&language=en&sortBy=publishedAt&pageSize=100"
        f"&apiKey={API_KEY}"
    )

    r = requests.get(url)

    print("Status:", r.status_code)

    if r.status_code != 200:
        print("Error:", r.text)
        continue

    data = r.json()

    if "articles" in data:
        for a in data["articles"]:
            all_news.append({
                "Date": a["publishedAt"][:10],
                "Ticker": ticker,
                "Headline": a["title"]
            })

    time.sleep(1)

df = pd.DataFrame(all_news)
df.to_csv("data/raw/news/news_data.csv", index=False)

print("DONE")
print(df.shape)