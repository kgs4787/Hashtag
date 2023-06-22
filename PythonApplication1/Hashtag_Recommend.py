import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
dataset = pd.read_csv('Top_hashtag.csv')

def extract_hashtags(text, dataset):
    hashtags = []
    stop_words = set(stopwords.words('english'))  # English stopwords

    # Text preprocessing: lowercase, remove special characters, etc.
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Extract hashtags matching the extracted words
    for hashtag in dataset['Hashtag']:
        if hashtag.lower() in filtered_tokens:
            hashtags.append(hashtag)

    return hashtags

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    sentiment = max(sentiment_scores, key=sentiment_scores.get)
    return sentiment

# Get user input
post_content = input("Enter your post content: ")

# Extract hashtags
hashtags = extract_hashtags(post_content, dataset)

# Top hashtags based on Likes
top_likes_hashtags = dataset.loc[dataset['Hashtag'].isin(hashtags)].sort_values('Likes', ascending=False).head(5)

# Print the results
if hashtags:
    print("Extracted hashtags from the post:")
    for hashtag in hashtags:
        print(hashtag)

    post_sentiment = analyze_sentiment(post_content)

    print("Post sentiment:", post_sentiment)

if not top_likes_hashtags.empty:
    print("Top 5 hashtags among the extracted hashtags based on Likes:")
    for index, row in top_likes_hashtags.iterrows():
        hashtag = row['Hashtag']
        posts = row['Posts']
        likes = row['Likes']
        print(f"{hashtag} (Number of posts: {posts}, Average of likes: {likes})")
else:
    print("No matching hashtags found in the post.")
