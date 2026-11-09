
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sampleSubmission.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2          

TARGET = "count"

LEAKY_COLUMNS = ["casual", "registered"]

CATEGORICAL_COLUMNS = ["season", "weather", "hour", "month", "dayofweek"]
NUMERIC_COLUMNS = ["temp", "atemp", "humidity", "windspeed"]
BINARY_COLUMNS = ["holiday", "workingday", "is_weekend", "year"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
