import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["product_id", "name", "category", "price"])
    df = df[df["price"] > 0]
    df = df[df["stock"] >= 0]
    df = df[
        (df["rating"] >= 0) &
        (df["rating"] <= 5)
    ]
    return df

def split_data(df: pd.DataFrame):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

def create_preprocessor():
    categorical_features = ["brand"]
    numerical_features = ["price", "stock", "rating"]
    preprocessor = ColumnTransformer(
        transformers = [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numerical_features)
        ]
    )
    return preprocessor