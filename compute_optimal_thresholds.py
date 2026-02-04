"""
Script to compute optimal thresholds for signal model predictions
using BERT model predictions (binarized via BERT_THRESHOLDS) as ground truth.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, f1_score, precision_recall_curve
from utils.constants import BERT_THRESHOLDS
import json
import argparse


def get_pattern_name(column_name: str) -> str:
    """Extract pattern name from column name by removing suffix."""
    if column_name.endswith('_bert_model'):
        return column_name[:-len('_bert_model')]
    elif column_name.endswith('_sig_model'):
        return column_name[:-len('_sig_model')]
    return column_name


def get_bert_threshold(pattern_name: str) -> float:
    """Get the BERT threshold for a given pattern name."""
    if pattern_name in BERT_THRESHOLDS:
        return BERT_THRESHOLDS[pattern_name].get('threshold', 0.5)
    return 0.5  # Default threshold if not found


def compute_optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """
    Compute optimal threshold using Youden's J statistic (sensitivity + specificity - 1).
    Returns (optimal_threshold, best_youden_j)
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # No positive or no negative samples
        return 0.5, 0.0
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    return thresholds[best_idx], youden_j[best_idx]


def compute_optimal_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """
    Compute optimal threshold that maximizes F1 score.
    Returns (optimal_threshold, best_f1)
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5, 0.0
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Compute F1 for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def process_file(input_path: str, model_name: str, verbose: bool = True):
    """
    Process a single probability file and return results.
    
    Args:
        input_path: Path to the input CSV file
        model_name: Name of the model (e.g., 'wcr', 'efficientnet')
        verbose: Whether to print progress
    
    Returns:
        results: Dictionary of threshold results per pattern
        binary_df: DataFrame with binarized predictions
    """
    if verbose:
        print(f"\nProcessing {model_name} from {input_path}...")
    
    df = pd.read_csv(input_path)
    if verbose:
        print(f"Loaded {len(df)} rows")
    
    # Identify bert_model and sig_model columns
    bert_columns = [col for col in df.columns if col.endswith('_bert_model')]
    sig_columns = [col for col in df.columns if col.endswith('_sig_model')]
    
    if verbose:
        print(f"Found {len(bert_columns)} BERT model columns")
        print(f"Found {len(sig_columns)} signal model columns")
    
    results = {}
    binary_data = {'file_name': df['file_name'].values}
    
    for bert_col in bert_columns:
        pattern_name = get_pattern_name(bert_col)
        sig_col = f"{pattern_name}_sig_model"
        
        if sig_col not in df.columns:
            if verbose:
                print(f"Warning: No matching signal column for {bert_col}")
            continue
        
        # Get BERT threshold and create ground truth
        bert_threshold = get_bert_threshold(pattern_name)
        y_true = (df[bert_col] >= bert_threshold).astype(int).values
        y_prob = df[sig_col].values
        
        # Store binarized BERT predictions
        binary_data[f"{pattern_name}_bert_binary"] = y_true
        
        # Compute optimal thresholds
        threshold_youden, youden_j = compute_optimal_threshold_youden(y_true, y_prob)
        threshold_f1, best_f1 = compute_optimal_threshold_f1(y_true, y_prob)
        
        # Binarize signal model predictions using optimal thresholds
        sig_binary_youden = (y_prob >= threshold_youden).astype(int)
        sig_binary_f1 = (y_prob >= threshold_f1).astype(int)
        binary_data[f"{pattern_name}_{model_name}_binary_youden"] = sig_binary_youden
        binary_data[f"{pattern_name}_{model_name}_binary_f1"] = sig_binary_f1
        
        # Compute agreement metrics
        n_positive_bert = y_true.sum()
        n_positive_sig_youden = sig_binary_youden.sum()
        n_positive_sig_f1 = sig_binary_f1.sum()
        
        # F1 with computed thresholds
        f1_youden = f1_score(y_true, sig_binary_youden, zero_division=0)
        f1_optimal = f1_score(y_true, sig_binary_f1, zero_division=0)
        
        results[pattern_name] = {
            'bert_threshold': float(bert_threshold),
            'optimal_threshold_youden': float(threshold_youden),
            'optimal_threshold_f1': float(threshold_f1),
            'youden_j_statistic': float(youden_j),
            'best_f1_score': float(best_f1),
            'f1_with_youden_threshold': float(f1_youden),
            'f1_with_f1_threshold': float(f1_optimal),
            'n_positive_bert': int(n_positive_bert),
            'n_positive_sig_youden': int(n_positive_sig_youden),
            'n_positive_sig_f1': int(n_positive_sig_f1),
            'prevalence': float(n_positive_bert / len(df))
        }
        
        if verbose:
            print(f"{pattern_name}:")
            print(f"  BERT threshold: {bert_threshold:.3f}, Positives: {n_positive_bert}")
            print(f"  Optimal threshold (Youden): {threshold_youden:.4f}, F1: {f1_youden:.4f}")
            print(f"  Optimal threshold (F1): {threshold_f1:.4f}, F1: {f1_optimal:.4f}")
    
    binary_df = pd.DataFrame(binary_data)
    return results, binary_df


def main():
    parser = argparse.ArgumentParser(description='Compute optimal thresholds for ECG predictions')
    parser.add_argument('--combine', action='store_true', 
                        help='Process both WCR and EfficientNet files and combine results')
    args = parser.parse_args()
    
    if args.combine:
        # Process both files and combine
        print("="*60)
        print("Processing WCR model...")
        print("="*60)
        wcr_results, wcr_binary = process_file(
            '/app/outputs/wcr_CLSA_F1_diagnosis_probabilities.csv',
            'wcr'
        )
        
        print("\n" + "="*60)
        print("Processing EfficientNet model...")
        print("="*60)
        eff_results, eff_binary = process_file(
            '/app/outputs/efficientnetv2_CLSA_F1_diagnosis_probabilities.csv',
            'efficientnet'
        )
        
        # Combine binary predictions (merge on file_name)
        # Get bert_binary columns from wcr (they should be the same)
        bert_cols = [c for c in wcr_binary.columns if c.endswith('_bert_binary')]
        wcr_sig_cols = [c for c in wcr_binary.columns if '_wcr_binary_youden' in c]  # Only Youden
        eff_sig_cols = [c for c in eff_binary.columns if '_efficientnet_binary_youden' in c]  # Only Youden
        
        # Start with file_name and bert_binary columns from wcr
        combined_df = wcr_binary[['file_name'] + bert_cols + wcr_sig_cols].copy()
        
        # Add efficientnet columns
        for col in eff_sig_cols:
            combined_df[col] = eff_binary[col].values
        
        # Rename columns: bert_binary -> bert_groundtruth, remove _youden suffix
        rename_map = {}
        for col in combined_df.columns:
            if col.endswith('_bert_binary'):
                rename_map[col] = col.replace('_bert_binary', '_bert_groundtruth')
            elif col.endswith('_wcr_binary_youden'):
                rename_map[col] = col.replace('_wcr_binary_youden', '_wcr_binary')
            elif col.endswith('_efficientnet_binary_youden'):
                rename_map[col] = col.replace('_efficientnet_binary_youden', '_efficientnet_binary')
        combined_df = combined_df.rename(columns=rename_map)
        
        # Save combined predictions
        combined_df.to_csv('/app/outputs/clsa_ecg_predictions_F1.csv', index=False)
        print(f"\nSaved combined predictions to /app/outputs/clsa_ecg_predictions_F1.csv")
        print(f"Total columns: {len(combined_df.columns)}")
        
        # Save thresholds for both models
        all_results = {
            'wcr': wcr_results,
            'efficientnet': eff_results
        }
        with open('/app/outputs/optimal_thresholds_combined.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("Saved combined thresholds to /app/outputs/optimal_thresholds_combined.json")
        
        # Create comparison summary
        summary_data = []
        for pattern in wcr_results.keys():
            if pattern in eff_results:
                summary_data.append({
                    'pattern': pattern,
                    'bert_threshold': wcr_results[pattern]['bert_threshold'],
                    'wcr_threshold_f1': wcr_results[pattern]['optimal_threshold_f1'],
                    'wcr_f1': wcr_results[pattern]['f1_with_f1_threshold'],
                    'efficientnet_threshold_f1': eff_results[pattern]['optimal_threshold_f1'],
                    'efficientnet_f1': eff_results[pattern]['f1_with_f1_threshold'],
                    'prevalence': wcr_results[pattern]['prevalence']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('wcr_f1', ascending=False)
        summary_df.to_csv('/app/outputs/threshold_comparison.csv', index=False)
        print("Saved comparison summary to /app/outputs/threshold_comparison.csv")
        
        # Print summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Average F1 (WCR): {summary_df['wcr_f1'].mean():.4f}")
        print(f"Average F1 (EfficientNet): {summary_df['efficientnet_f1'].mean():.4f}")
        print(f"\nTop 10 patterns:")
        print(summary_df.head(10).to_string(index=False))
        
    else:
        # Original behavior - process WCR only
        results, binary_df = process_file(
            '/app/outputs/wcr_CLSA_F1_diagnosis_probabilities.csv',
            'sig'
        )
        
        # Save results
        print("\nSaving results...")
        
        # Save thresholds as JSON
        with open('/app/outputs/optimal_thresholds.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Saved optimal thresholds to /app/outputs/optimal_thresholds.json")
        
        # Save binarized predictions
        binary_df.to_csv('/app/outputs/binarized_predictions.csv', index=False)
        print("Saved binarized predictions to /app/outputs/binarized_predictions.csv")
        
        # Create summary table
        summary_data = []
        for pattern, metrics in results.items():
            summary_data.append({
                'pattern': pattern,
                'bert_threshold': metrics['bert_threshold'],
                'optimal_threshold_youden': metrics['optimal_threshold_youden'],
                'optimal_threshold_f1': metrics['optimal_threshold_f1'],
                'f1_with_youden': metrics['f1_with_youden_threshold'],
                'f1_with_f1_opt': metrics['f1_with_f1_threshold'],
                'prevalence': metrics['prevalence']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('f1_with_f1_opt', ascending=False)
        summary_df.to_csv('/app/outputs/threshold_summary.csv', index=False)
        print("Saved summary to /app/outputs/threshold_summary.csv")
        
        # Print overall statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Average F1 (Youden): {summary_df['f1_with_youden'].mean():.4f}")
        print(f"Average F1 (F1-opt): {summary_df['f1_with_f1_opt'].mean():.4f}")
        print(f"\nTop 10 patterns by F1 score:")
        print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

