import logging
import os
from datetime import datetime

# --- where logs live: a 'logs' folder at the REPO ROOT (not the current dir) ---
# __file__ = .../uav-gps-localization/pipeline/logger.py  -> go up two levels.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# If the orchestrator (run_pipeline.py) set one shared path, every stage uses it,
# so the whole pipeline run lands in ONE file. Standalone runs make their own.
LOG_FILE_PATH = os.environ.get("PIPELINE_LOG_FILE")
if not LOG_FILE_PATH:
    LOG_FILE = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s",
)


if __name__ == "__main__":
    pass
