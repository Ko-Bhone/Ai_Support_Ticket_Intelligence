import pandas as pd

def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["product_id", "name", "category", "price"])
    df = df[df["stock"] >= 0]
    df = df[
        (df["rating"] >=0) &
        (df["rating"] <= 5)]

    return df

