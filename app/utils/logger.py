import logging
import os
from logging.handlers import RotatingFileHandler

class AppLogger:
    """
    Class to set up and manage the application logger with rotating file handling.
    """

    def __init__(self, log_directory='logs', log_file='app.log', max_bytes=1000000, backup_count=5):
        """
        Initialize the logger with rotating file handler and stream handler.
        
        Args:
            log_directory (str): Directory where logs will be stored.
            log_file (str): Log file name.
            max_bytes (int): Maximum size of log file before rotation.
            backup_count (int): Number of backup log files to keep.
        """
        self.log_directory = log_directory
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self.logger = self._setup_logger()

    def _setup_logger(self):
        """
        Set up the logger with a rotating file handler and stream handler.

        Returns:
            logger (logging.Logger): Configured logger instance.
        """
        # Ensure log directory exists
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        # Configure rotating file handler
        log_path = os.path.join(self.log_directory, self.log_file)
        handler = RotatingFileHandler(log_path, maxBytes=self.max_bytes, backupCount=self.backup_count)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Set up logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Avoid adding handlers multiple times
        if not logger.handlers:
            logger.addHandler(handler)
            logger.addHandler(logging.StreamHandler())  # Add console output

        # Prevent logging propagation to the root logger
        logger.propagate = False

        return logger

    def get_logger(self):
        """
        Return the configured logger instance.

        Returns:
            logger (logging.Logger): The application logger instance.
        """
        return self.logger
