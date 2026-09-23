import pandas as pd

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.isnull().sum()