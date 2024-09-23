import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    # Load general configurations from environment variables
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')  # Default to 'development' if not set
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')  # Default key for dev, should be changed in prod
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    ES_HOST = os.getenv('ES_HOST', 'localhost:9200')
    ES_INDEX = os.getenv('ES_INDEX', 'reddit_posts')

    # Ensure required variables are not missing
    @classmethod
    def validate_required_vars(cls):
        required_vars = {
            "REDDIT_CLIENT_ID": cls.REDDIT_CLIENT_ID,
            "REDDIT_CLIENT_SECRET": cls.REDDIT_CLIENT_SECRET,
            "REDDIT_USER_AGENT": cls.REDDIT_USER_AGENT,
            "SECRET_KEY": cls.SECRET_KEY
        }
        for var_name, value in required_vars.items():
            if value is None:
                raise ValueError(f"Required environment variable {var_name} is not set.")

    @staticmethod
    def load_yaml_config(yaml_file_name):
        """Load YAML file configuration."""
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
        yaml_file_path = os.path.join(base_dir, 'utils', 'config', yaml_file_name)  # Path relative to config.py

        try:
            with open(yaml_file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML configuration file '{yaml_file_path}' not found.")
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error loading YAML file '{yaml_file_path}': {exc}")
