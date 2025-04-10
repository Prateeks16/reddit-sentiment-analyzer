import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import praw
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from transformers import pipeline  # Added for sentiment analysis
import requests  # Added for News API integration

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load environment variables from .env file
load_dotenv()

# Cache the sentiment classifier to avoid reloading
@st.cache_resource
def load_sentiment_classifier():
    with st.spinner("Loading sentiment analysis model (this may take a moment on first run)..."):
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", framework="pt")

# Initialize with error handling
try:
    sentiment_classifier = load_sentiment_classifier()
except Exception as e:
    st.error(f"Failed to load sentiment model: {e}")
    st.write("Ensure PyTorch is installed (e.g., 'pip install torch') and your internet is stable.")
    st.stop()

# Fetch Reddit posts using credentials from .env
@st.cache_data
def fetch_reddit_posts(subreddit, query, count, start_date, end_date):
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')
    
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent) 
    if subreddit == 'all':
        all_posts = list(reddit.subreddit('all').search(query, limit=count))
    else:
        all_posts = list(reddit.subreddit(subreddit).search(query, limit=count))
    
    total_fetched = len(all_posts)
    
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
    start_timestamp = start_dt.timestamp()
    end_timestamp = end_dt.timestamp()
    
    filtered_posts = [
        (post.title + " " + post.selftext, post.created_utc)
        for post in all_posts
        if start_timestamp <= post.created_utc <= end_timestamp
    ]
    
    return filtered_posts, total_fetched

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+|@\w+|#|[^A-Za-z\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

# Sentiment analysis function using Hugging Face Transformers
def analyze_sentiment(text):
    text = text[:512]
    result = sentiment_classifier(text)[0]
    label = result['label']
    if label == 'LABEL_2':  # Positive
        return 'positive'
    elif label == 'LABEL_0':  # Negative
        return 'negative'
    else:  # LABEL_1 is Neutral
        return 'neutral'

# Fetch news articles using News API
def fetch_news_articles(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=5"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles
    except requests.RequestException as e:
        st.error(f"Error fetching news articles: {e}")
        return []

# Initialize session state for tracking analysis
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Streamlit app layout
st.title("Reddit Sentiment Analysis 🗣️")

# Show sentiment analysis description and how-to-use guide only if not analyzed
if not st.session_state.analyzed:
    st.markdown("""
    ### What is Sentiment Analysis?
    Sentiment analysis is a natural language processing (NLP) technique used to determine the emotional tone behind a piece of text. It classifies opinions as **positive**, **neutral**, or **negative**, helping to understand public sentiment on topics like news, products, or events. In this app, we use a pre-trained RoBERTa model from Hugging Face to analyze Reddit posts, giving you insights into how people feel about your chosen topic.

    ### How to Use This App
    1. **Set Up in the Sidebar**:
       - **Subreddit**: Choose a subreddit (e.g., "news" or "all") to search.
       - **Search Query**: Enter a keyword (e.g., "election") to filter posts.
       - **Post Count**: Slide to select how many posts to analyze (10–100).
       - **Date Range**: Pick start and end dates for the posts (defaults to the last 30 days).
    2. **Start Analysis**: Click the "Analyze" button in the sidebar.
    3. **Explore Results**: After analysis, this section will disappear, and you'll see:
       - Charts showing sentiment distribution and trends.
       - Word clouds for positive and negative sentiments.
       - Sample posts for each sentiment type.
       - Related news articles (if News API is configured).
       - A CSV download option for the data.
    4. **Adjust and Repeat**: Change settings and analyze again as needed!
    """)

# Sidebar for user inputs
st.sidebar.header("Search Settings")
subreddit = st.sidebar.selectbox(
    "Select Subreddit",
    ['all', 'news', 'worldnews', 'politics', 'technology', 'science'],
    help="Choose a subreddit or 'all' for a broad search."
)
query = st.sidebar.text_input("Search Query", "election")

st.sidebar.header("Analysis Settings")
post_count = st.sidebar.slider("Maximum Number of Posts", 10, 100, 50)
start_date = st.sidebar.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", value=datetime.now().date())
st.sidebar.write("Note: Fewer posts may be analyzed if limited by the date range or API.")

# Analyze button
if st.sidebar.button("Analyze"):
    st.session_state.analyzed = True  # Set flag to hide description and guide
    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
    else:
        with st.spinner(f"Fetching up to {post_count} posts from r/{subreddit} for '{query}'..."):
            try:
                data, total_fetched = fetch_reddit_posts(subreddit, query, post_count, start_date, end_date)
            except Exception as e:
                st.error(f"Error fetching posts: {e}")
            else:
                if total_fetched == 0:
                    st.error("No posts found for the given query and subreddit. Try a different query.")
                elif len(data) == 0:
                    st.error("No posts found within the selected date range. Adjust the dates.")
                else:
                    st.write(f"Fetched {total_fetched} posts, {len(data)} within the date range.")
                    with st.spinner("Processing posts and analyzing sentiment..."):
                        df = pd.DataFrame(data, columns=['text', 'timestamp'])
                        df['cleaned_text'] = df['text'].apply(preprocess_text)
                        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
                        df['date'] = df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).date())
                    
                    # Sentiment Distribution (Bar Chart)
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = df['sentiment'].value_counts()
                    st.bar_chart(sentiment_counts)
                    
                    # Sentiment Distribution (Pie Chart)
                    st.subheader("Sentiment Distribution (Pie Chart)")
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Sentiment Trend Over Time
                    st.subheader("Sentiment Trend Over Time")
                    trend = df.groupby('date')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
                    st.line_chart(trend)
                    
                    # Word Clouds for Positive and Negative Sentiments
                    st.subheader("Word Clouds")
                    for sentiment in ['positive', 'negative']:
                        words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
                        if words:
                            with st.spinner(f"Generating word cloud for {sentiment} sentiment..."):
                                wordcloud = WordCloud(width=800, height=400).generate(words)
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                ax.set_title(f"{sentiment.capitalize()} Words")
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            st.write(f"No {sentiment} posts to display.")
                    
                    # Sample Posts for Each Sentiment
                    st.subheader("Sample Posts")
                    for sentiment in ['positive', 'neutral', 'negative']:
                        with st.expander(f"{sentiment.capitalize()} Posts"):
                            sample_posts = df[df['sentiment'] == sentiment]['text'].head(5).tolist()
                            if sample_posts:
                                for post in sample_posts:
                                    st.write(post)
                                    st.write("---")
                            else:
                                st.write(f"No {sentiment} posts found.")
                    
                    # Summary Statistics
                    st.subheader("Summary")
                    total_posts = len(df)
                    positive = (df['sentiment'] == 'positive').sum()
                    neutral = (df['sentiment'] == 'neutral').sum()
                    negative = (df['sentiment'] == 'negative').sum()
                    st.write(f"Total posts analyzed: {total_posts}")
                    st.write(f"Positive: {positive} ({positive / total_posts:.2%})")
                    st.write(f"Neutral: {neutral} ({neutral / total_posts:.2%})")
                    st.write(f"Negative: {negative} ({negative / total_posts:.2%})")
                    
                    # Fetch and Display News Articles
                    st.subheader("Related News Articles")
                    news_api_key = os.getenv('NEWS_API_KEY')
                    if news_api_key:
                        articles = fetch_news_articles(query, news_api_key)
                        if articles:
                            with st.expander("View News Articles"):
                                for article in articles:
                                    st.write(f"**{article['title']}**")
                                    st.write(f"Source: {article['source']['name']}")
                                    st.write(f"Published: {article['publishedAt']}")
                                    st.write(article['description'])
                                    st.write(f"[Read more]({article['url']})")
                                    st.write("---")
                        else:
                            st.write("No news articles found for this query.")
                    else:
                        st.write("News API key not found. Please configure it in the .env file.")
                    
                    # Export Option
                    st.subheader("Export Results")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"reddit_sentiment_{subreddit}_{query}.csv",
                        mime="text/csv",
                    )
