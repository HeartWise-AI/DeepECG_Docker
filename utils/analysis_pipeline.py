
import json
import pandas as pd
from models import ModelFactory, BertClassifier

def convert_to_df(df_path: str) -> pd.DataFrame:
    df = pd.read_parquet(df_path)
    return df

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)

def save_to_json(data: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

class AnalysisPipeline:
    @staticmethod
    def run_analysis(
        df: pd.DataFrame, 
        signal_processing_model: ModelFactory, 
        classification_model: BertClassifier
    ) -> dict:
        pass