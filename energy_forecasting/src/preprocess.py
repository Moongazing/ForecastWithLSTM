
import pandas as pd
from src.config import DATA_PATH, TARGET_COLUMN

def load_and_preprocess():
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        infer_datetime_format=True,
        na_values=["?"],
        low_memory=False
    )

    df.set_index("datetime", inplace=True)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df = df[[TARGET_COLUMN]].dropna()
    df = df.resample("1H").mean()  
    df = df.fillna(method="ffill")
    
    return df
