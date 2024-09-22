from app.models.bert_model import classify_intent
from app.services.reddit_service import fetch_reddit_posts
from app.services.elasticsearch_service import index_documents

def classify_reddit_posts(posts, top_k=3):
    """
    Classify the intent of Reddit posts.
    
    Args:
        posts (list): A list of Reddit posts.
        top_k (int): The number of top intents to return.
    
    Returns:
        list: A list of dictionaries containing post details and classified intents.
    """
    classified_posts = []
    for post in posts:
        content = post.title + " " + post.selftext
        intents = classify_intent(content, top_k=top_k)
        classified_post = {
            'id': post.id,
            'title': post.title,
            'content': post.selftext,
            'intents': intents,
            'created_utc': post.created_utc,
            'author': post.author.name if post.author else None,
            'subreddit': post.subreddit.display_name
        }
        classified_posts.append(classified_post)
    return classified_posts

def fetch_classify_and_index_posts(subreddit_name, limit=10, top_k=3):
    """
    Fetch Reddit posts, classify their intent, and index them into Elasticsearch.
    
    Args:
        subreddit_name (str): The name of the subreddit to fetch posts from.
        limit (int): The number of posts to fetch.
        top_k (int): The number of top intents to return.
    """
    # Fetch Reddit posts
    posts = fetch_reddit_posts(subreddit_name, limit)

    # Classify intents of the posts
    classified_posts = classify_reddit_posts(posts, top_k=top_k)

    # Index classified posts into Elasticsearch
    index_documents(classified_posts)