from app.clients.reddit_client import RedditClient

class PostClassifier:
    """
    Class to handle the classification of Reddit posts.
    """

    def __init__(self, reddit_client, tokenizer, top_k=3):
        """
        Initialize the classifier with the number of top topics to return, RedditClient instance, and BertTokenizer instance.
        
        Args:
            reddit_client (RedditClient): An instance of RedditClient to fetch Reddit posts.
            tokenizer (BertTokenizer): An instance of BertTokenizer to tokenize the content.
            top_k (int): The number of top topics to return for each post.
        """
        self.reddit_client = reddit_client
        self.tokenizer = tokenizer
        self.top_k = top_k

    def classify_reddit_posts(self, posts):
        """
        Classify the topics of Reddit posts.
        
        Args:
            posts (list): A list of Reddit posts.
        
        Returns:
            list: A list of dictionaries containing post details and classified topics.
        """
        classified_posts = []

        for post in posts:
            # Handle post content: title + content, check for None or empty values
            content = post.title if post.selftext is None else f"{post.title} {post.selftext}"
            
            # Classify the post's topics using the model
            topics = self.tokenizer.classify_topic(content, top_k=self.top_k)
            
            # Structure the topic data (ensure clear separation of topic and score)
            classified_topics = [
                {"name": topic[0], "score": topic[1]} for topic in topics
            ]
            
            # Convert post details into a structured dictionary
            classified_post = {
                'id': post.id,
                'title': post.title,
                'content': post.selftext if post.selftext else "",  # Handle empty content
                'topics': classified_topics,  # Structured topics with names and scores
                'created_utc': post.created_utc,  # Keep the original epoch time for consistency
                'author': post.author.name if post.author else "Unknown",  # Handle missing author
                'subreddit': post.subreddit.display_name if post.subreddit else "Unknown"  # Handle missing subreddit
            }
            
            # Add classified post to the result list
            classified_posts.append(classified_post)
        
        return classified_posts


    def fetch_and_classify_posts(self, subreddit_name, limit=10):
        """
        Fetch posts from a specific subreddit and classify them.
        
        Args:
            subreddit_name (str): The name of the subreddit to fetch posts from.
            limit (int): The number of posts to fetch.
        
        Returns:
            list: A list of classified Reddit posts.
        """
        posts = self.reddit_client.fetch_reddit_posts(subreddit_name, limit)
        return self.classify_reddit_posts(posts)