# ================================
# IMPORT REQUIRED MODULES
# ================================

import logging          # Used to create logs (info, warning, error etc.)
import os               # Used for file/folder path handling
from datetime import datetime   # Used to get current date & time


# ================================
# CREATE LOG FILE NAME
# ================================

# Example output:
# 05_13_2026_01_45_10.log

# strftime converts current time into formatted string
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


# ================================
# CREATE LOG FOLDER PATH
# ================================

# os.getcwd() → gives current working directory
# "logs" → folder name
# LOG_FILE → log file name

# Example:
# D:/project/logs/05_13_2026_01_45_10.log

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)


# ================================
# CREATE DIRECTORY IF NOT EXISTS
# ================================

# Creates folders automatically
# exist_ok=True → avoids error if folder already exists

os.makedirs(logs_path, exist_ok=True)


# ================================
# FULL LOG FILE PATH
# ================================

# Final path for log file storage

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# ================================
# CONFIGURE LOGGING
# ================================

logging.basicConfig(

    # Where logs will be stored
    filename=LOG_FILE_PATH,

    # Format of log message
    # Example:
    # [2026-05-13 01:50:10] 45 mymodule - INFO - message

    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # Minimum level to track
    # INFO means:
    # INFO, WARNING, ERROR, CRITICAL will be stored

    level=logging.INFO,
)


if __name__ == "__main__":
    logging.info("Logging has started")