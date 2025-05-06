import logging
import os
from datetime import datetime
from typing import Any


class CustomLogger:
    def __init__(
        self, name: str = "ThyroidScan", log_level: Any = logging.INFO, log_to_file: bool = True, log_dir: str = "logs"
    ) -> None:
        """
        Initializing the logger

        :param name: logger name
        :param log_level: logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
        :param log_to_file: need to log in to a file
        :param log_dir: dir for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger
