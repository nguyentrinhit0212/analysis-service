from flask import Flask
from app.utils.logger import logger
from dotenv import load_dotenv
import os

def create_app():
    # Load environment variables from .env file
    load_dotenv()
    logger.info("Environment variables loaded from .env file.")

    app = Flask(__name__)
    logger.info("Flask application instance created.")

    # Load configurations from environment variables
    app.config['FLASK_ENV'] = os.getenv('FLASK_ENV')
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['REDDIT_CLIENT_ID'] = os.getenv('REDDIT_CLIENT_ID')
    app.config['REDDIT_CLIENT_SECRET'] = os.getenv('REDDIT_CLIENT_SECRET')
    app.config['REDDIT_USER_AGENT'] = os.getenv('REDDIT_USER_AGENT')
    app.config['ES_HOST'] = os.getenv('ES_HOST')
    logger.info("Configurations loaded from environment variables.")

    # Register blueprints
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    logger.info("Blueprint 'main' registered.")

    return app