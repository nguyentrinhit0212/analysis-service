import schedule
import time
import threading
from app.clients.reddit_client import RedditClient
from app.classifiers.post_classifier import PostClassifier
from app.models.bert_model import BertModel
from app.utils.logger import AppLogger
from threading import Lock

class RedditPostScheduler:
    """
    Class to handle the scheduling of tasks for fetching, classifying,
    and indexing Reddit posts into Elasticsearch.
    """

    def __init__(self, subreddit_name, elastic_client, limit=100, top_k=3, interval_minutes=10):
        """
        Initialize the scheduler with the required parameters.
        """
        self.logger = AppLogger().get_logger()
        self.subreddit_name = subreddit_name
        self.elastic_client = elastic_client
        self.limit = limit
        self.top_k = top_k
        self.interval_minutes = interval_minutes
        self.lock = Lock()  # Add a threading lock to prevent multiple executions

        # Initialize RedditClient, BertModel, and PostClassifier
        self.reddit_client = RedditClient()
        self.bert_model = BertModel('yahoo_answers/checkpoint-78750')
        self.post_classifier = PostClassifier(self.reddit_client, self.bert_model, top_k=self.top_k)

    def fetch_classify_and_index_posts(self):
        """
        Fetch Reddit posts, classify their topics, and index them into Elasticsearch.
        """
        # Check if another thread is already running this method
        if self.lock.locked():
            self.logger.info("Job already running, skipping this execution.")
            return

        with self.lock:  # Acquire the lock to ensure only one execution
            try:
                self.logger.info(f"Fetching {self.limit} posts from subreddit: {self.subreddit_name}")
                posts = self.reddit_client.fetch_reddit_posts(self.subreddit_name, self.limit)

                self.logger.info("Classifying fetched posts.")
                classified_posts = self.post_classifier.classify_reddit_posts(posts)

                self.logger.info("Indexing classified posts into Elasticsearch.")
                self.elastic_client.index_documents('reddit_posts', classified_posts)

                self.logger.info("Fetch, classify, and index cycle completed successfully.")
            except Exception as e:
                self.logger.error(f"Error in fetch_classify_and_index_posts: {e}")

    def schedule_reddit_post_fetch(self):
        """
        Schedule a cron-like job to fetch, classify, and index Reddit posts at regular intervals.
        """
        self.logger.info(f"Scheduling Reddit post fetch every {self.interval_minutes} minutes.")
        schedule.every(self.interval_minutes).minutes.do(self.fetch_classify_and_index_posts)

        while True:
            schedule.run_pending()
            time.sleep(1)

    def start_scheduler(self):
        """
        Start the scheduler in a separate thread to continuously fetch and classify posts.
        """
        scheduler_thread = threading.Thread(
            target=self.schedule_reddit_post_fetch,
            daemon=True  # Daemonize thread to exit when main thread exits
        )
        scheduler_thread.start()
        self.logger.info("Scheduler thread started.")
