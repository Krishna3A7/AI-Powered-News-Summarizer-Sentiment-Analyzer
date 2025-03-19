#importing the required libraries
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import streamlit as st

#Fetch News Articles
def get_news_articles(keyword):
    url = f"https://news.google.com/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return ["Failed to fetch news"]
    
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article", limit=5)
    
    news_data = []
    for article in articles:
        title = article.find("a").text
        link = "https://news.google.com" + article.find("a")["href"][1:]
        news_data.append({"title": title, "link": link})
    
    return news_data

#Summarize new with LLM
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

# Sentiment Analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive 😊"
    elif score <= -0.05:
        return "Negative 😞"
    else:
        return "Neutral 😐"

# Streamlit UI
st.title("📰 AI-Powered News Summarizer & Sentiment Analyzer")

keyword = st.text_input("Enter a keyword for news search:", "AI")

if st.button("Fetch News"):
    articles = get_news_articles(keyword)
    
    for article in articles:
        st.subheader(article["title"])
        st.write(f"[Read More]({article['link']})")
        summary = summarize_text(article["title"]) # Summarize news article
        st.write("**Summary:**", summary)
        sentiment = analyze_sentiment(summary) # Sentiment Analysis
        st.write("**Sentiment:**", sentiment)
        st.write("---")