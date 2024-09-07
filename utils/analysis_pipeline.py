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
from models import HeartWiseModelFactory, BertClassifier
from utils.constants import ECG_CATEGORIES, ECG_PATTERNS
from utils.ecg_signal_processor import ECGSignalProcessor


def compute_metrics(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    categories_gt = {category: [] for category in ECG_CATEGORIES}
    categories_pred = {category: [] for category in ECG_CATEGORIES}
    prevalence_gt = {}
    prevalence_pred = {}
    total_samples = len(df_gt)
    
    for col in df_gt.columns:
        sum_gt = df_gt[col].sum()
        sum_pred = df_pred[col].sum()
        prevalence_gt[col] = (sum_gt / total_samples) * 100        
        prevalence_pred[col] = (sum_pred / total_samples) * 100        
        for category in ECG_CATEGORIES:
            if col in ECG_CATEGORIES[category]:
                if sum_gt == 0:
                    continue
                categories_gt[category].append(df_gt[col])
                categories_pred[category].append(df_pred[col])
                prevalence_gt[category] = prevalence_gt.get(category, 0) + sum_gt
                prevalence_pred[category] = prevalence_pred.get(category, 0) + sum_pred

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
        cat_auc_scores = []
        cat_auprc_scores = []
        cat_f1_scores = []
        sum_gt = 0
        sum_pred = 0
        for col_gt, col_pred in zip(categories_gt[category], categories_pred[category]):
            sum_gt += col_gt.sum()
            sum_pred += col_pred.sum()
            cat_auc_scores.append(roc_auc_score(col_gt, col_pred))
            cat_auprc_scores.append(average_precision_score(col_gt, col_pred))
            cat_f1_scores.append(f1_score(col_gt, col_pred > 0.5))

        prevalence_gt[category] = sum_gt / len(categories_gt[category])
        prevalence_pred[category] = sum_pred / len(categories_pred[category])
        
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
                int(np.array(categories_pred[category]).ravel() > 0.5), 
                average='micro'
            ),
            "prevalence_gt": float(prevalence_gt[category]),
            "prevalence_pred": float(prevalence_pred[category])
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
            "prevalence_gt": float(prevalence_gt[col]),
            "prevalence_pred": float(prevalence_pred[col])
        }
        auc_scores.append(metrics[col]['auc'])
        auprc_scores.append(metrics[col]['auprc'])
        f1_scores.append(metrics[col]['f1'])

    metrics['dataset'] = {
        'macro_auc': np.mean(auc_scores),
        'macro_auprc': np.mean(auprc_scores),
        'macro_f1': np.mean(f1_scores),
        'micro_auc': roc_auc_score(df_gt[:, 1:].values.ravel(), df_pred[:, 1:].values.ravel(), average='micro'),
        'micro_auprc': average_precision_score(df_gt[:, 1:].values.ravel(), df_pred[:, 1:].values.ravel(), average='micro'),
        'micro_f1': f1_score(df_gt[:, 1:].values.ravel(), int(df_pred[:, 1:].values.ravel() > 0.5), average='micro')
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