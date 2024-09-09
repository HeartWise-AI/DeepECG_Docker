import os
import csv
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from utils.files_handler import XMLProcessor, ECGFileHandler
from data.project_dataset import create_dataloader
from models import HeartWiseModelFactory
from utils.constants import ECG_CATEGORIES, ECG_PATTERNS
from utils.ecg_signal_processor import ECGSignalProcessor


def compute_metrics(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    categories_gt = {category: [] for category in ECG_CATEGORIES}
    categories_pred = {category: [] for category in ECG_CATEGORIES}
    prevalence_classes_gt = {}
    prevalence_classes_pred = {}
    total_samples = len(df_gt)
    
    # Gather data for each category and compute prevalence per class
    for col in df_gt.columns:
        sum_gt = df_gt[col].sum()
        sum_pred = df_pred[col].sum()
        prevalence_classes_gt[col] = (sum_gt / total_samples) * 100        
        prevalence_classes_pred[col] = (sum_pred / total_samples) * 100        
        for category in ECG_CATEGORIES:
            if col in ECG_CATEGORIES[category]:
                if sum_gt == 0:
                    continue
                categories_gt[category].append(df_gt[col])
                categories_pred[category].append(df_pred[col])

    # Compute prevalence per category
    df_prevalence_gt = pd.DataFrame(0, index=df_gt.index, columns=ECG_CATEGORIES.keys())
    df_prevalence_pred = pd.DataFrame(0, index=df_gt.index, columns=ECG_CATEGORIES.keys())
    for category, cols in ECG_CATEGORIES.items():
        df_prevalence_gt[category] = (df_gt[cols] == 1).any(axis=1).astype(int)
        df_prevalence_pred[category] = (df_pred[cols] > 0.5).any(axis=1).astype(int)
    total_samples = len(df_gt)
    prevalence_categories_gt = df_prevalence_gt.sum() / total_samples
    prevalence_categories_pred = df_prevalence_pred.sum() / total_samples
                    
    metrics = {}
    for category in ECG_CATEGORIES:
        if not categories_gt[category] or not categories_pred[category]:
            metrics[category] = {
                "macro_auc": np.nan,
                "macro_auprc": np.nan,
                "micro_auc": np.nan,
                "micro_auprc": np.nan,
                "macro_f1": np.nan,
                "micro_f1": np.nan,
                "prevalence_gt": np.nan,
                "prevalence_pred": np.nan
            }
            continue
        
        # Compute per-category metrics
        cat_auc_scores = []
        cat_auprc_scores = []
        cat_f1_scores = []
        for col_gt, col_pred in zip(categories_gt[category], categories_pred[category]):
            cat_auc_scores.append(roc_auc_score(col_gt, col_pred))
            cat_auprc_scores.append(average_precision_score(col_gt, col_pred))
            cat_f1_scores.append(f1_score(col_gt, col_pred > 0.5))
        
        # Store Category Metrics
        metrics[category] = {
            "macro_auc": np.mean(cat_auc_scores),
            "macro_auprc": np.mean(cat_auprc_scores),
            "macro_f1": np.mean(cat_f1_scores),
            "micro_auc": roc_auc_score(
                np.array(categories_gt[category]).ravel(), 
                np.array(categories_pred[category]).ravel(), 
                average='micro'
            ),
            "micro_auprc": average_precision_score(
                np.array(categories_gt[category]).ravel(), 
                np.array(categories_pred[category]).ravel(), 
                average='micro'
            ),
            "micro_f1": f1_score(
                np.array(categories_gt[category]).ravel(), 
                np.array(categories_pred[category]).ravel() > 0.5, 
                average='micro'
            ),
            "prevalence_gt": float(prevalence_categories_gt[category]),
            "prevalence_pred": float(prevalence_categories_pred[category])
        }

    # Compute per-class metrics and collect data for each pattern
    auc_scores = []
    auprc_scores = []
    f1_scores = []
    for col in ECG_PATTERNS:
        sum_gt = df_gt[col].sum()
        if sum_gt == 0:
            metrics[col] = {
                "auc": np.nan,
                "auprc": np.nan,
                "f1": np.nan,
                "prevalence_gt": np.nan,
                "prevalence_pred": np.nan
            }
            continue
        metrics[col] = {
            "auc": roc_auc_score(df_gt[col], df_pred[col]),
            "auprc": average_precision_score(df_gt[col], df_pred[col]),
            "f1": f1_score(df_gt[col], df_pred[col] > 0.5),
            "prevalence_gt": float(prevalence_classes_gt[col]),
            "prevalence_pred": float(prevalence_classes_pred[col])
        }
        auc_scores.append(metrics[col]['auc'])
        auprc_scores.append(metrics[col]['auprc'])
        f1_scores.append(metrics[col]['f1'])

    metrics['dataset'] = {
        'macro_auc': np.mean(auc_scores),
        'macro_auprc': np.mean(auprc_scores),
        'macro_f1': np.mean(f1_scores),
        'micro_auc': roc_auc_score(df_gt.values.ravel(), df_pred.values.ravel(), average='micro'),
        'micro_auprc': average_precision_score(df_gt.values.ravel(), df_pred.values.ravel(), average='micro'),
        'micro_f1': f1_score(df_gt.values.ravel(), df_pred.values.ravel() > 0.5, average='micro')
    }
        
    return metrics

class AnalysisPipeline:
    @staticmethod
    def preprocess_data(df: pd.DataFrame, output_folder: str):
        # Generate XML
        xml_processor = XMLProcessor() 
        processed_files, ecg_signals = xml_processor.process_batch(
            df=df,
            num_workers=16
        )        
        print(f"Processed {len(processed_files)} files.")
        
        # Save report
        xml_processor.save_report(output_folder=output_folder)
        
        # Process ECG signals
        ecg_signal_processor = ECGSignalProcessor()
        cleaned_ecg_signals = ecg_signal_processor.clean_and_process_ecg_leads(input_data=ecg_signals, max_workers=16)
        
        np.save(os.path.join(output_folder, 'cleaned_ecgs.npy'), cleaned_ecg_signals)
        os.makedirs("./tmp", exist_ok=True)
        for i, file in tqdm(enumerate(processed_files), total=len(processed_files)):
            ECGFileHandler.save_ecg_signal(
                ecg_signal=cleaned_ecg_signals[i],
                filename=f"{file}"
            )
        
        
    @staticmethod
    def run_analysis(
        df: pd.DataFrame, 
        batch_size: int,
        diagnosis_classifier_device: int,
        signal_processing_device: int,
        signal_processing_model_name: str, 
        diagnosis_classifier_model_name: str,
        hugging_face_api_key: str
    ) -> dict:
        # Load models
        diagnosis_classifier_model = HeartWiseModelFactory.create_model(
            {
                'model_name': diagnosis_classifier_model_name,
                'map_location': torch.device(diagnosis_classifier_device),
                'hugging_face_api_key': hugging_face_api_key
            }
        )
        
        signal_processing_model = HeartWiseModelFactory.create_model(
            {
                'model_name': signal_processing_model_name,
                'map_location': torch.device(signal_processing_device),
                'hugging_face_api_key': hugging_face_api_key
            }
        )        
        
        # Compute bert diagnoses predictions
        predictions = []
        ground_truth = []
        dataloader = create_dataloader(df, batch_size=batch_size, shuffle=False)
        for diagnosis, ecg_tensor in tqdm(dataloader, total=len(dataloader)):
            diag_prob = diagnosis_classifier_model(diagnosis)
            sig_prob = signal_processing_model(ecg_tensor)
            for i in range(len(diag_prob)):
                ground_truth.append(torch.where(diag_prob[i] > 0.5, 1, 0).detach().cpu().numpy())
                predictions.append(sig_prob[i].detach().cpu().numpy())

        # Compute and return metrics
        return compute_metrics(
            df_gt=pd.DataFrame(ground_truth, columns=ECG_PATTERNS), 
            df_pred=pd.DataFrame(predictions, columns=ECG_PATTERNS)
        )