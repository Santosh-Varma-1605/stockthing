import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import os

# --------------------------
# SETUP
# --------------------------
os.makedirs("data/processed/sentiment", exist_ok=True)

model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# --------------------------
# LOAD NEWS DATA
# --------------------------
df = pd.read_csv("data/raw/news/news_data.csv")

# Basic cleaning
df = df.dropna(subset=["Headline", "Date", "Ticker"])

# --------------------------
# SENTIMENT FUNCTION
# --------------------------
def get_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    # labels: [negative, neutral, positive]
    score = probs[0][2] - probs[0][0]

    return score.item()

# --------------------------
# APPLY MODEL
# --------------------------
tqdm.pandas()
df["Sentiment"] = df["Headline"].progress_apply(get_sentiment)

# --------------------------
# AGGREGATE
# --------------------------
agg = df.groupby(["Date", "Ticker"]).agg({
    "Sentiment": "count"
}).reset_index()

agg.columns = ["Date", "Ticker", "Sentiment_Count"]

# --------------------------
# SENTIMENT SPIKE
# --------------------------
agg["Sentiment_Spike"] = (
    agg.groupby("Ticker")["Sentiment_Count"]
    .transform(lambda x: x / x.rolling(10).mean())
)

# --------------------------
# SAVE
# --------------------------
agg.to_csv("data/processed/sentiment/sentiment_features.csv", index=False)

print("Sentiment dataset created!")
print(agg.head())