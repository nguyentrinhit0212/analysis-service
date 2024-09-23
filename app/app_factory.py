import threading
from flask import Flask
from app.config import Config
from app.services.reddit_scheduler import RedditPostScheduler
from app.clients.elasticsearch_client import ElasticClient
from app.controllers import PostAnalyzer
from app.utils.logger import AppLogger

class AppFactory:
    def __init__(self):
        self.app = None
        self.logger = AppLogger().get_logger()
        self.scheduler_started = False  # Ensure scheduler runs only once
        self.logger.info("AppFactory instance created.")

    def create_app(self):
        """
        Create and configure the Flask application.
        """
        self.app = Flask(__name__)
        self.app.config.from_object(Config)
        self.logger.info("Flask application instance created.")

        # Load index settings and mappings from a YAML file
        self.es_index_settings = Config.load_yaml_config('reddit_topic_analize_es_index_config.yaml')

        # Initialize ElasticService
        self.elastic_client = ElasticClient()

        # Ensure Elasticsearch index exists
        self.create_elasticsearch_index()

        # Initialize RedditPostScheduler
        self.reddit_scheduler = RedditPostScheduler(
            subreddit_name='AskReddit',
            elastic_client=self.elastic_client,
            limit=100,
            top_k=1,
            interval_minutes=10
        )

        # Start the scheduler in the background
        self.run_scheduler_in_background()

        # Register routes
        self.register_routes()

        return self.app

    def create_elasticsearch_index(self):
        """
        Ensure the Elasticsearch index is created based on the configurations.
        """
        self.elastic_client.create_index(self.app.config['ES_INDEX'], self.es_index_settings)
        self.logger.info(f"Elasticsearch index '{self.app.config['ES_INDEX']}' ensured.")

    def start_scheduler(self):
        """
        Start the Reddit post-fetching scheduler in a separate thread.
        Ensure this is done only once.
        """
        if not self.scheduler_started:
            self.logger.info("Starting the Reddit posts fetching scheduler.")
            self.reddit_scheduler.start_scheduler()
            self.scheduler_started = True
            self.logger.info("Scheduler for Reddit posts is now running.")
        else:
            self.logger.warning("Scheduler is already running.")

    def run_scheduler_in_background(self):
        """
        Run the scheduler in a separate background thread to avoid blocking the Flask app.
        """
        if not self.scheduler_started:
            self.logger.info("Starting the scheduler in a background thread.")
            scheduler_thread = threading.Thread(target=self.start_scheduler)
            scheduler_thread.daemon = True  # Ensure the thread exits when the main program exits
            scheduler_thread.start()
            self.logger.info("Scheduler is now running in the background.")
        else:
            self.logger.warning("Scheduler is already running in the background.")

    def register_routes(self):
        """
        Register Flask routes.
        """
        post_analyzer = PostAnalyzer()
        self.app.register_blueprint(post_analyzer.get_blueprint())
