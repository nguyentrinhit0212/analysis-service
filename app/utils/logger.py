import logging
import os
from logging.handlers import RotatingFileHandler

# Tạo thư mục logs nếu chưa tồn tại
if not os.path.exists('logs'):
    os.makedirs('logs')

# Cấu hình RotatingFileHandler
handler = RotatingFileHandler('logs/app.log', maxBytes=1000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Thiết lập logging
logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])

# Tạo logger cho ứng dụng
logger = logging.getLogger(__name__)
