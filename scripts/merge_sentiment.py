import pandas as pd
import os

os.makedirs("data/final", exist_ok=True)

# --------------------------
# LOAD DATA
# --------------------------
stock_df = pd.read_csv("data/final/sp500_top10_dataset.csv")
sent_df = pd.read_csv("data/processed/sentiment/sentiment_features.csv")

# --------------------------
# MERGE
# --------------------------
df = pd.merge(stock_df, sent_df, on=["Date", "Ticker"], how="left")

# --------------------------
# SHIFT (CRITICAL)
# --------------------------
df["Sentiment_Spike"] = df.groupby("Ticker")["Sentiment_Spike"].shift(1)

# --------------------------
# HANDLE MISSING
# --------------------------
df["Sentiment_Spike"] = df["Sentiment_Spike"].fillna(1)

# --------------------------
# FINAL CLEAN
# --------------------------
df = df.dropna()

# --------------------------
# SAVE
# --------------------------
df.to_csv("data/final/sp500_with_sentiment.csv", index=False)

print("Final dataset with sentiment created!")
print(df.shape)