from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.data.preprocessing import create_preprocessor

def create_classification_pipeline():

    preprocessor = create_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline