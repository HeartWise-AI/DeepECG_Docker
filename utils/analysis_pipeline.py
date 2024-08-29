import os
import csv
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from models import ModelFactory, BertClassifier
from data.project_dataset import create_dataloader
from utils.constants import ECG_CATEGORIES, ECG_PATTERNS
from sklearn.metrics import roc_auc_score, average_precision_score

def convert_to_df(df_path: str) -> pd.DataFrame:
    df = pd.read_parquet(df_path)
    return df

def save_to_csv(metrics: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))

    # Open the file and create a CSV writer
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for metric in metrics:
            for value in metrics[metric]:
                writer.writerow([metric, value, metrics[metric][value]])

def save_to_json(data: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def compute_metrics(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    categories_gt = {category: [] for category in ECG_CATEGORIES}
    categories_pred = {category: [] for category in ECG_CATEGORIES}
    for col in df_gt.columns[1:]:
        for category in ECG_CATEGORIES:
            if col in ECG_CATEGORIES[category]:
                if df_gt[col].sum() == 0:
                    continue
                categories_gt[category].append(df_gt[col])
                categories_pred[category].append(df_pred[col])

    metrics = {}
    for category in ECG_CATEGORIES:
        if not categories_gt[category] or not categories_pred[category]:
            metrics[category] = {
                "macro_auc": np.nan,
                "macro_auprc": np.nan,
                "micro_auc": np.nan,
                "micro_auprc": np.nan
            }
            continue
        cat_auc_scores = []
        cat_auprc_scores = []
        for col_gt, col_pred in zip(categories_gt[category], categories_pred[category]):
            cat_auc_scores.append(roc_auc_score(col_gt, col_pred))
            cat_auprc_scores.append(average_precision_score(col_gt, col_pred))

        metrics[category] = {
            "macro_auc": np.mean(cat_auc_scores),
            "macro_auprc": np.mean(cat_auprc_scores),
            "micro_auc": roc_auc_score(
                np.array(categories_gt[category]).ravel(), 
                np.array(categories_pred[category]).ravel(), 
                average='micro'
            ),
            "micro_auprc": average_precision_score(
                np.array(categories_gt[category]).ravel(), 
                np.array(categories_pred[category]).ravel(), 
                average='micro'
            )
        }

    # Compute per-class metrics and collect data for each pattern
    auc_scores = []
    auprc_scores = []
    for col in ECG_PATTERNS:
        class_sum = df_gt[col].sum()
        if class_sum == 0:
            continue
        metrics[col] = {
            "auc": roc_auc_score(df_gt[col], df_pred[col]),
            "auprc": average_precision_score(df_gt[col], df_pred[col])
        }
        auc_scores.append(metrics[col]['auc'])
        auprc_scores.append(metrics[col]['auprc'])

    metrics['dataset'] = {
        'macro_auc': np.mean(auc_scores),
        'macro_auprc': np.mean(auprc_scores),
        'micro_auc': roc_auc_score(df_gt.iloc[:, 1:].values.ravel(), df_pred.iloc[:, 1:].values.ravel(), average='micro'),
        'micro_auprc': average_precision_score(df_gt.iloc[:, 1:].values.ravel(), df_pred.iloc[:, 1:].values.ravel(), average='micro')          
    }
        
    return metrics

class AnalysisPipeline:
    @staticmethod
    def run_analysis(
        df_path: str, 
        signal_processing_model: ModelFactory, 
        diagnosis_classifier_model: BertClassifier
    ) -> dict:
        # Load data
        df = pd.read_parquet(df_path)
        
        batch_size = 32
        
        # Compute bert diagnoses predictions
        predictions = []
        ground_truth = []
        dataloader = create_dataloader(df, batch_size=batch_size)
        for diagnosis, npy_path, labels in tqdm(dataloader, total=len(dataloader)):
            # diag_prob = diagnosis_classifier_model(diagnosis)
            sig_prob = signal_processing_model(npy_path)
            for i in range(len(sig_prob)):
                # ground_truth.append(torch.where(diag_prob[i] > 0.5, 1, 0).detach().cpu().numpy())
                predictions.append(sig_prob[i].detach().cpu().numpy())

                ground_truth.append(labels[i].detach().cpu().numpy())                

        # Compute and return metrics
        return compute_metrics(
            df_gt=pd.DataFrame(ground_truth, columns=ECG_PATTERNS), df_pred=pd.DataFrame(predictions, columns=ECG_PATTERNS))