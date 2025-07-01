
from pathlib import Path

DATA_PATH = Path("data/household_power_consumption.txt")
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"
TARGET_COLUMN = "Global_active_power"

SEQUENCE_LENGTH = 24  
TEST_RATIO = 0.2
BATCH_SIZE = 64
EPOCHS = 10
