import praw
from app.config import Config

class RedditClient:
    """
    Class to handle interactions with Reddit API.
    """

    def __init__(self):
        """
        Initialize Reddit client using credentials from configuration.
        """
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT
        )

    def fetch_reddit_posts(self, subreddit_name, limit=10):
        """
        Fetch Reddit posts from a specific subreddit.
        
        Args:
            subreddit_name (str): The name of the subreddit to fetch posts from.
            limit (int): The number of posts to fetch.
        
        Returns:
            list: A list of Reddit posts.
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        return subreddit.new(limit=limit)
