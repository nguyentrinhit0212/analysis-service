import praw
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def fetch_reddit_posts(subreddit_name, limit=10):
    """
    Fetch Reddit posts from a specific subreddit.
    
    Args:
        subreddit_name (str): The name of the subreddit to fetch posts from.
        limit (int): The number of posts to fetch.
    
    Returns:
        list: A list of Reddit posts.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.new(limit=limit)
    return posts