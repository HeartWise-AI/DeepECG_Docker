import torch
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    roc_curve
)
from utils.files_handler import XMLProcessor, ECGFileHandler
from data.project_dataset import create_dataloader
from models import HeartWiseModelFactory
from utils.constants import (
    ECG_CATEGORIES, 
    ECG_PATTERNS, 
    BERT_THRESHOLDS, 
    WCR_COLUMN_CONVERSION, 
    PTBXL_POWER_RATIO
)
from utils.ecg_signal_processor import ECGSignalProcessor
from utils.log_config import get_logger
from utils.error_collector import collect

logger = get_logger(__name__)

def compute_best_threshold(df_gt_col: pd.Series, df_pred_col: pd.Series) -> float:
    """
    Compute the best threshold using the Youden Index.

    Args:
        df_gt_col (pd.Series): Ground truth values for a specific column.
        df_pred_col (pd.Series): Predicted values for a specific column.

    Returns:
        float: The best threshold value.
    """
    fpr, tpr, roc_thresholds = roc_curve(df_gt_col, df_pred_col)
    
    # Compute Youden Index
    youden_index = tpr - fpr
    best_threshold = roc_thresholds[np.argmax(youden_index)]
    
    return float(best_threshold) # convert to float instead of numpy.float32

def compute_metrics_binary(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics for binary ECG classification.

    Args:
        df_gt (pd.DataFrame): Ground truth DataFrame with one column.
        df_pred (pd.DataFrame): Predicted probabilities DataFrame with one column.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """    
    if df_gt.shape[1] != 1 or df_pred.shape[1] != 1:
        raise ValueError("Both df_gt and df_pred must have exactly one column each for binary classification.")

    # Extract the single column
    gt = df_gt.iloc[:, 0]
    pred = df_pred.iloc[:, 0]

    # Initialize metrics dictionary
    metrics = {
        "results": {
            "auc": np.nan,
            "auprc": np.nan,
            "f1": np.nan,
            "threshold": np.nan,
            "prevalence_gt %": np.nan,
            "prevalence_pred %": np.nan
        }
    }

    # Check if there are positive samples in ground truth
    if gt.sum() == 0:
        logger.warning("No positive samples in ground truth; metrics will be NaN.")
        return metrics

    try:
        # Compute ROC AUC
        metrics["results"]["auc"] = roc_auc_score(gt, pred)

        # Compute Average Precision (AUPRC)
        metrics["results"]["auprc"] = average_precision_score(gt, pred)

        # Compute Best Threshold
        best_threshold = compute_best_threshold(gt, pred)
        metrics["results"]["threshold"] = best_threshold

        # Compute F1 Score
        predictions_binary = (pred >= best_threshold).astype(int)
        metrics["results"]["f1"] = f1_score(gt, predictions_binary)

        # Compute Prevalence in Ground Truth
        metrics["results"]["prevalence_gt %"] = (gt.sum() / len(gt)) * 100

        # Compute Prevalence in Predictions
        metrics["results"]["prevalence_pred %"] = (predictions_binary.sum() / len(pred)) * 100

    except Exception as e:
        logger.error("Binary classification metrics failed: %s", e, exc_info=True)
    logger.debug("Computed metrics: %s", metrics)
    return metrics

def compute_metrics(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics for ECG classification.

    Args:
        df_gt (pd.DataFrame): Ground truth DataFrame.
        df_pred (pd.DataFrame): Predicted probabilities DataFrame.

    Returns:
        dict: A dictionary containing evaluation metrics for each category and column.
    """    
    # initialize metrics dictionary
    metrics = {}
    for cat in ECG_CATEGORIES:
        metrics[cat] = {
            "macro_auc": np.nan,
            "macro_auprc": np.nan,
            "macro_f1": np.nan,
            "micro_auc": np.nan,
            "micro_auprc": np.nan,
            "micro_f1": np.nan,
            "threshold": np.nan,
            "prevalence_gt %": np.nan,
            "prevalence_pred %": np.nan
        }
    for col in df_gt.columns:
        metrics[col] = {
            "auc": np.nan,
            "auprc": np.nan,
            "f1": np.nan,
            "threshold": np.nan,
            "prevalence_gt %": np.nan,
            "prevalence_pred %": np.nan
        }

    # Compute category metrics
    for category in ECG_CATEGORIES:
        # Get category columns
        category_columns = [
            col for col in ECG_CATEGORIES[category]
            if df_gt[col].sum() > 0 # filter out columns with no ground truth
        ]
                        
        # Skip category if no columns have ground truth
        if not category_columns:
            continue
            
        # Aggregate ground truth and predictions for the category
        category_gt = df_gt[category_columns]
        category_pred = df_pred[category_columns]

        # Compute macro auc and auprc metrics
        cat_auc_scores = []
        cat_auprc_scores = []
        macro_f1_scores = []
        for col in category_columns:
            # Compute metrics for each column
            col_auc = roc_auc_score(category_gt[col], category_pred[col])
            col_auprc = average_precision_score(category_gt[col], category_pred[col])
            col_threshold = compute_best_threshold(category_gt[col], category_pred[col])
            col_f1 = f1_score(category_gt[col], category_pred[col] >= col_threshold)
            metrics[col] = {
                "auc": col_auc,
                "auprc": col_auprc,
                "threshold": col_threshold,
                "f1": col_f1,
                "prevalence_gt %": category_gt[col].sum() / len(df_gt) * 100,
                "prevalence_pred %": (category_pred[col] >= col_threshold).sum() / len(df_pred) * 100,
            }
            
            # Append metrics to category metrics
            cat_auc_scores.append(col_auc)
            cat_auprc_scores.append(col_auprc)
            macro_f1_scores.append(col_f1)
            
        # Compute macro metrics
        cat_macro_auc = np.mean(cat_auc_scores)
        cat_macro_auprc = np.mean(cat_auprc_scores)
        cat_macro_f1 = np.mean(macro_f1_scores)
                    
        # Compute micro metrics
        ravel_categories_gt = category_gt.values.ravel()
        ravel_categories_pred = category_pred.values.ravel()
        micro_auc = roc_auc_score(
            ravel_categories_gt, 
            ravel_categories_pred, 
            average='micro'
        )
        micro_auprc = average_precision_score(
            ravel_categories_gt, 
            ravel_categories_pred, 
            average='micro'
        )
        best_micro_threshold = compute_best_threshold(ravel_categories_gt, ravel_categories_pred)
        cat_micro_f1 = f1_score(ravel_categories_gt, ravel_categories_pred >= best_micro_threshold, average='micro')                    
        
        # Compute Category Prevalence
        cat_prevalence_gt = ravel_categories_gt.sum() / len(ravel_categories_gt) * 100
        cat_prevalence_micro = (ravel_categories_pred >= best_micro_threshold).sum() / len(ravel_categories_pred) * 100
                        
        # Store Category Metrics
        metrics[category] = {
            "macro_auc":  cat_macro_auc,
            "macro_auprc": cat_macro_auprc,
            "macro_f1": cat_macro_f1,
            "micro_auc": micro_auc,
            "micro_auprc": micro_auprc,
            "micro_f1": cat_micro_f1,
            "threshold": best_micro_threshold,
            "prevalence_gt %": cat_prevalence_gt,
            "prevalence_pred %": cat_prevalence_micro
        }
        
    return metrics
 
class AnalysisPipeline:
    """Orchestrates ECG preprocessing (NPY/XML) and model-based analysis with metrics."""

    @staticmethod
    def save_and_preprocess_data(
        df: pd.DataFrame,
        output_folder: str,
        preprocessing_folder: str,
        preprocessing_n_workers: int,
        errors: list[str] | None = None,
    ) -> pd.DataFrame | None:
        """
        Preprocess ECG files from df (NPY or XML), scale and clean signals, and save as .base64.

        Processes in batches; failed files or batches are skipped and optionally reported via errors.
        Returns None if no data was successfully processed.
        """
        ecg_signal_processor = ECGSignalProcessor()
        batch_size = 10000
        total_batches = (len(df) + batch_size - 1) // batch_size
        processed_df = pd.DataFrame()

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            
            logger.info("Processing batch %d/%d (%d records)", batch_idx + 1, total_batches, end_idx - start_idx)
            
            try:
                # Get batch of data
                batch_df = df.iloc[start_idx:end_idx].copy()
                
                # Process based on file type
                if batch_df['ecg_path'].iloc[0].endswith('.npy'):
                    ecgs = []
                    # Process NPY files
                    for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Loading NPY files"):
                        try:
                            lead_array = np.load(row['ecg_path'])

                            if np.isnan(lead_array).any():
                                continue
                            
                            file_id = os.path.basename(row['ecg_path']).replace(".npy", "")
                            
                            # Shape corrections
                            if lead_array.shape[-1] == 1:
                                lead_array = lead_array.squeeze(-1)
                            if lead_array.shape[0] == 12:  # transpose if needed
                                lead_array = lead_array.transpose(1, 0)
                                
                            # Handle different lengths
                            if lead_array.shape[0] != 2500:
                                if lead_array.shape[0] < 2500:
                                    collect(
                                        errors, "preprocessing",
                                        "Skipped file (signal too short)",
                                        f"file_id={file_id} length={lead_array.shape[0]}",
                                    )
                                    continue
                                else:
                                    step = lead_array.shape[0] // 2500
                                    lead_array = lead_array[::step, :]
                                    
                            if lead_array.shape[1] != 12:
                                collect(
                                    errors, "preprocessing",
                                    "Skipped file (wrong lead count)",
                                    f"file_id={file_id} leads={lead_array.shape[1]}",
                                )
                                continue
                                
                            new_path = os.path.join(preprocessing_folder, f"{file_id}.base64")
                            batch_df.at[index, 'ecg_path'] = new_path
                            ecgs.append([new_path, lead_array])
                            
                        except Exception as e:
                            collect(errors, "preprocessing", "Failed to process file", f"path={row['ecg_path']} error={e}")
                            continue
                            
                    ecg_signals_df = pd.DataFrame(ecgs, columns=['ecg_path', 'ecg_signal'])
                    
                else:
                    # Process XML files
                    try:
                        xml_processor = XMLProcessor()
                        batch_df, ecg_signals_df = xml_processor.process_batch(
                            df=batch_df,
                            num_workers=preprocessing_n_workers,
                            preprocessing_folder=preprocessing_folder
                        )
                        
                        # Save XML processing report for this batch
                        xml_processor.save_report(
                            output_folder=os.path.join(output_folder, f'batch_{batch_idx + 1}')
                        )
                        
                    except Exception as e:
                        collect(errors, "preprocessing", f"XML processing failed for batch {batch_idx + 1}", str(e))
                        continue
                if len(ecg_signals_df) == 0:
                    collect(errors, "preprocessing", f"No valid signals in batch {batch_idx + 1}", None)
                    continue
                logger.info("Scaling ECG signals...")
                scaled_signals_df = ecg_signal_processor.scale_ecg_signals(
                    df=ecg_signals_df, 
                    power_ratio=PTBXL_POWER_RATIO
                )
                
                logger.info("Processing ECG signals...")
                cleaned_signals_df = ecg_signal_processor.clean_and_process_ecg_leads(
                    df=scaled_signals_df,
                    max_workers=preprocessing_n_workers
                )
                
                logger.info("Saving processed signals...")
                for _, row in tqdm(cleaned_signals_df.iterrows(), total=len(cleaned_signals_df), desc="Saving signals"):
                    ECGFileHandler.save_ecg_signal(
                        ecg_signal=row['ecg_signal'],
                        filename=row['ecg_path']
                    )
                
                # Append processed batch
                processed_df = pd.concat([processed_df, batch_df], ignore_index=True)
                
                # Clear memory
                del ecg_signals_df, scaled_signals_df, cleaned_signals_df, batch_df
                
            except Exception as e:
                collect(errors, "preprocessing", f"Batch {batch_idx + 1} failed", str(e))
                continue
        if len(processed_df) == 0:
            collect(errors, "preprocessing", "No data was successfully processed", None)
            return None
        logger.info("Preprocessing complete: %d files", len(processed_df))
        return processed_df
              
    @staticmethod
    def run_analysis(
        df: pd.DataFrame,
        batch_size: int,
        diagnosis_classifier_device: int,
        signal_processing_device: int,
        signal_processing_model_name: str,
        diagnosis_classifier_model_name: str,
        hugging_face_api_key: str,
        errors: list[str] | None = None,
    ) -> tuple[dict | None, pd.DataFrame | None]:
        """
        Run inference and compute metrics. On failure, if errors is provided, append and return (None, None).
        """
        try:
            return AnalysisPipeline._run_analysis_impl(
                df=df,
                batch_size=batch_size,
                diagnosis_classifier_device=diagnosis_classifier_device,
                signal_processing_device=signal_processing_device,
                signal_processing_model_name=signal_processing_model_name,
                diagnosis_classifier_model_name=diagnosis_classifier_model_name,
                hugging_face_api_key=hugging_face_api_key,
            )
        except Exception as e:
            collect(errors, "analysis", "Run failed", f"model={signal_processing_model_name} error={e}")
            return None, None

    @staticmethod
    def _run_analysis_impl(
        df: pd.DataFrame,
        batch_size: int,
        diagnosis_classifier_device: int,
        signal_processing_device: int,
        signal_processing_model_name: str,
        diagnosis_classifier_model_name: str,
        hugging_face_api_key: str,
    ) -> tuple[dict, pd.DataFrame]:
        """Internal implementation: load models, run inference, return (metrics, df_probabilities)."""
        signal_processing_model = HeartWiseModelFactory.create_model(
            {
                'model_name': signal_processing_model_name,
                'map_location': torch.device(signal_processing_device),
                'hugging_face_api_key': hugging_face_api_key
            }
        )
        
        if "77" in signal_processing_model_name:
            # Load models
            diagnosis_classifier_model = HeartWiseModelFactory.create_model(
                {
                    'model_name': diagnosis_classifier_model_name,
                    'map_location': torch.device(diagnosis_classifier_device),
                    'hugging_face_api_key': hugging_face_api_key
                }
            )
            
            # Compute bert diagnoses predictions
            predictions = []
            ground_truth = []
            probabities_rows = []
            logger.info("Creating dataloader (batch_size=%d)...", batch_size)
            dataloader = create_dataloader(df, batch_size=batch_size, shuffle=False)
                        
            for batch in tqdm(dataloader, total=len(dataloader)):
                diagnosis = batch['diagnosis']
                ecg_tensor = batch['ecg_signal'].to(signal_processing_device)
                file_name = batch['file_name']

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
                if "wcr" in signal_processing_model_name:
                    sig_prob = sig_prob[:, WCR_COLUMN_CONVERSION]
                        
                for i in range(len(diag_binary)):
                    ground_truth.append(diag_binary[i].detach().cpu().numpy())
                    predictions.append(sig_prob[i].detach().cpu().numpy())
                    probabities_rows.append([file_name[i]] + list(diag_prob[i].detach().cpu().numpy()) + list(sig_prob[i].detach().cpu().numpy()))
                            
            bert_columns = [f"{pattern}_bert_model" for pattern in ECG_PATTERNS]
            sig_columns = [f"{pattern}_sig_model" for pattern in ECG_PATTERNS]
            columns = ['file_name'] + bert_columns + sig_columns
            df_probabilities = pd.DataFrame(probabities_rows, columns=columns)

            # Skip metrics computation if only one ECG (requires multiple samples for AUC, F1, etc.)
            if len(ground_truth) <= 1:
                logger.warning("Skipping metrics computation: only %d ECG(s) provided", len(ground_truth))
                return {}, df_probabilities

            return compute_metrics(
                df_gt=pd.DataFrame(ground_truth, columns=ECG_PATTERNS), 
                df_pred=pd.DataFrame(predictions, columns=ECG_PATTERNS)
            ), df_probabilities
        
        else:
            dataloader = create_dataloader(df, batch_size=batch_size, shuffle=False)
            predictions = []
            ground_truth = []
            probabilities_rows = []
            with torch.no_grad():
                for batch in tqdm(dataloader, total=len(dataloader)):
                    diagnosis = batch['diagnosis'].unsqueeze(-1)
                    ecg_tensor = batch['ecg_signal'].to(signal_processing_device)
                    file_name = batch['file_name']      

                    sig_prob = signal_processing_model(ecg_tensor)

                    for i in range(len(sig_prob)):
                        predictions.append(sig_prob[i].detach().cpu().numpy())
                        ground_truth.append(diagnosis[i].detach().cpu().numpy())
                        probabilities_rows.append([file_name[i]] + list(diagnosis[i].detach().cpu().numpy()) + list(sig_prob[i].detach().cpu().numpy()))

            df_probabilities = pd.DataFrame(probabilities_rows, columns=['file_name', 'ground_truth', 'predictions'])

            # Skip metrics computation if only one ECG (requires multiple samples for AUC, F1, etc.)
            if len(ground_truth) <= 1:
                logger.warning("Skipping metrics computation: only %d ECG(s) provided", len(ground_truth))
                return {}, df_probabilities

            return compute_metrics_binary(
                df_gt=pd.DataFrame(ground_truth, columns=["ground_truth"]), 
                df_pred=pd.DataFrame(predictions, columns=["predictions"])
            ), df_probabilities
