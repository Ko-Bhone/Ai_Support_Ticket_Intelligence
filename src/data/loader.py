import pandas as pd

def load_products(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Products Datasets Loaded")
    print(f"Shape: {df.shape}")

    return df

