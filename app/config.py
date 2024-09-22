from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

# Cấu hình kết nối Elasticsearch
ES_HOST = os.getenv('ES_HOST')