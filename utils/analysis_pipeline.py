import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from utils.files_handler import XMLProcessor, ECGFileHandler
from data.project_dataset import create_dataloader
from models import HeartWiseModelFactory
from utils.constants import ECG_CATEGORIES, ECG_PATTERNS, BERT_THRESHOLDS
from utils.ecg_signal_processor import ECGSignalProcessor


def compute_metrics(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    metrics = {}
    for category in ECG_CATEGORIES:
        category_columns = [
            col for col in ECG_CATEGORIES[category]
            if df_gt[col].sum() > 0
        ]
        
        if not category_columns:
            metrics[category] = {
                metric: np.nan for metric in [
                    "macro_auc", "macro_auprc", "micro_auc", "micro_auprc",
                    "macro_f1", "micro_f1", "prevalence_gt %", "prevalence_pred %"
                ]
            }
            continue

        category_gt, category_pred = zip(*[
            (df_gt[col], df_pred[col])
            for col in ECG_CATEGORIES[category]
            if df_gt[col].sum() > 0
        ])
        
        # Compute macro auc and auprc metrics
        cat_auc_scores = []
        cat_auprc_scores = []
        for col_gt, col_pred in zip(category_gt, category_pred):
            cat_auc_scores.append(roc_auc_score(col_gt, col_pred))
            cat_auprc_scores.append(average_precision_score(col_gt, col_pred))
        cat_macro_auc = np.mean(cat_auc_scores)
        cat_macro_auprc = np.mean(cat_auprc_scores)
                
        # Find best macro and micro f1 metrics
        ravel_categories_gt = np.array(category_gt).ravel()
        ravel_categories_pred = np.array(category_pred).ravel()
        best_macro_f1, best_micro_f1 = 0, 0
        macro_threshold, micro_threshold = 0.5, 0.5
        for threshold in np.arange(0, 1.01, 0.01):
            # Macro F1
            cat_f1_scores = [
                f1_score(col_gt, col_pred >= threshold) 
                for col_gt, col_pred in zip(category_gt, category_pred)
            ]
            macro_f1 = np.mean(cat_f1_scores)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                macro_threshold = threshold
            
            # Micro F1
            micro_f1 = f1_score(
                ravel_categories_gt, 
                ravel_categories_pred >= threshold, 
                average='micro'
            )
            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                micro_threshold = threshold
                        
        # Compute Category Prevalence                
        df_cat = pd.DataFrame(0, index=range(len(df_gt)), columns=[category]) 
        df_cat_macro = pd.DataFrame(0, index=range(len(df_gt)), columns=[category]) 
        df_cat_micro = pd.DataFrame(0, index=range(len(df_gt)), columns=[category]) 
        for col in ECG_CATEGORIES[category]:
            df_cat.loc[df_gt[col] == 1, category] = 1
            df_cat_macro.loc[(df_pred[col] >= macro_threshold).astype(int) == 1, category] = 1
            df_cat_micro.loc[(df_pred[col] >= micro_threshold).astype(int) == 1, category] = 1
                
        cat_prevalence_gt = float(
            (df_cat[category].sum() / len(df_cat)) * 100
        )
        cat_prevalence_macro = float(
            (df_cat_macro[category].sum() / len(df_cat_macro)) * 100
        )
        cat_prevalence_micro = float(
            (df_cat_micro[category].sum() / len(df_cat_micro)) * 100
        )
        
        # Store Category Metrics
        metrics[category] = {
            "macro_auc":  cat_macro_auc,
            "macro_auprc": cat_macro_auprc,
            "macro_f1": best_macro_f1,
            "macro_threshold": macro_threshold,
            "micro_auc": roc_auc_score(
                ravel_categories_gt, 
                ravel_categories_pred, 
                average='micro'
            ),
            "micro_auprc": average_precision_score(
                ravel_categories_gt, 
                ravel_categories_pred, 
                average='micro'
            ),
            "micro_f1": best_micro_f1,
            "micro_threshold": micro_threshold,
            "prevalence_gt %": cat_prevalence_gt,
            "prevalence_macro %": cat_prevalence_macro,
            "prevalence_micro %": cat_prevalence_micro
        }

    # Compute per-class metrics and collect data for each pattern
    for col in ECG_PATTERNS:
        sum_gt = df_gt[col].sum()
        if sum_gt == 0:
            metrics[col] = {
                "auc": np.nan,
                "auprc": np.nan,
                "f1": np.nan,
                "prevalence_gt %": np.nan,
                "prevalence_pred %": np.nan
            }
            continue
        
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0, 1.01, 0.01):
            f1 = f1_score(df_gt[col], df_pred[col] >= threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        prevalence_gt = float(
            (sum_gt / len(df_gt)) * 100
        )
        prevalence_pred = float(
            ((df_pred[col] >= best_threshold).astype(int).sum() / len(df_pred)) * 100
        )
            
        metrics[col] = {
            "auc": roc_auc_score(df_gt[col], df_pred[col]),
            "auprc": average_precision_score(df_gt[col], df_pred[col]),
            "f1": best_f1,
            "threshold": best_threshold,
            "prevalence_gt": prevalence_gt,
            "prevalence_pred": prevalence_pred
        }
        
    return metrics

class AnalysisPipeline:
    @staticmethod
    def preprocess_data(df: pd.DataFrame, output_folder: str) -> pd.DataFrame:
        # Generate XML
        xml_processor = XMLProcessor() 
        df, ecg_signals = xml_processor.process_batch(
            df=df,
            num_workers=16
        )        
        print(f"Processed {len(df)} files.")
        
        # Save report
        xml_processor.save_report(output_folder=output_folder)
        
        # Process ECG signals
        ecg_signal_processor = ECGSignalProcessor()
        cleaned_ecg_signals = ecg_signal_processor.clean_and_process_ecg_leads(input_data=ecg_signals, max_workers=16)
        
        print(f"Cleaned {len(cleaned_ecg_signals)} ecg signals.")
        
        for i, file in tqdm(enumerate(df['ecg_path']), total=len(df)):
            ECGFileHandler.save_ecg_signal(
                ecg_signal=cleaned_ecg_signals[i],
                filename=f"{file}"
            )
            
        return df
        
        
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
        probabities_rows = []
        dataloader = create_dataloader(df, batch_size=batch_size, shuffle=False)
        for diagnosis, ecg_tensor, file_name in tqdm(dataloader, total=len(dataloader)):
            # Create thresholds tensor
            current_batch_size = len(diagnosis)
            bert_thresholds_tensor = torch.zeros((current_batch_size, 77)).to(diagnosis_classifier_device)
            # Fill thresholds tensor
            for i, pattern in enumerate(ECG_PATTERNS):
                bert_thresholds_tensor[:, i] = BERT_THRESHOLDS[pattern]['threshold']

            # Compute and create diagnosis gt tensor
            diag_prob = diagnosis_classifier_model(diagnosis)
            diag_binary = torch.where(diag_prob >= bert_thresholds_tensor, 1, 0)
            
            # Append batch data 
            sig_prob = signal_processing_model(ecg_tensor)     
            for i in range(len(diag_prob)):
                ground_truth.append(diag_binary[i].detach().cpu().numpy())
                predictions.append(sig_prob[i].detach().cpu().numpy())
                probabities_rows.append([file_name[i]] + list(diag_prob[i].detach().cpu().numpy()) + list(sig_prob[i].detach().cpu().numpy()))
                        
        bert_columns = [f"{pattern}_bert_model" for pattern in ECG_PATTERNS]
        sig_columns = [f"{pattern}_sig_model" for pattern in ECG_PATTERNS]
        columns = ['file_name'] + bert_columns + sig_columns
        df_probabilities = pd.DataFrame(probabities_rows, columns=columns)

        # Compute and return metrics
        return compute_metrics(
            df_gt=pd.DataFrame(ground_truth, columns=ECG_PATTERNS), 
            df_pred=pd.DataFrame(predictions, columns=ECG_PATTERNS)
        ), df_probabilities