import os
import sys
from pathlib import Path
import logging

# Create log files in a "logs" folder
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    # Determine where logs go
    handlers=[
        logging.FileHandler(log_filepath),  # Write to file
        logging.StreamHandler(sys.stdout)  # Print in terminal
    ]
)
# Create a named logger can use a thorough project

""" DEBUG    |   Detailed diagnostic info
    INFO     |   Confirmation things are working
    WARNING  |   Something unexpected but not breaking
    ERROR    |   A function failed
    CRITICAL |   Program may crash"""

logger = logging.getLogger("cnnClassifierLogger")
