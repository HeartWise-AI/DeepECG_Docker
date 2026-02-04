import os
import sys
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", message=".*_register_pytree_node.*deprecated", category=FutureWarning)

from utils.constants import (
    Mode, 
    DIAGNOSIS_TO_FILE_COLUMNS,
    MODEL_MAPPING
)
from utils.parser import HearWiseArgs
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import (
    save_to_csv,
    save_json,
    read_api_key,
    save_df,
    load_df,
)
from utils.log_config import get_logger

logger = get_logger(__name__)

def set_up_directories(args: HearWiseArgs):
    """
    Set up necessary directories for output and preprocessing.

    This function creates the required output and preprocessing directories 
    based on the paths provided in the `args` object. If the directories 
    already exist, it will not raise an error.

    Args:
        args (HearWiseArgs): An instance containing configuration arguments, 
                             including `output_folder` and `preprocessing_folder`.

    Raises:
        OSError: If there is an error creating the directories due to permission issues 
                 or invalid paths.
    """    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    # Create preprocessing folder
    os.makedirs(args.preprocessing_folder, exist_ok=True)

def save_and_perform_preprocessing(args: HearWiseArgs, df: pd.DataFrame, errors: list[str] | None = None):
    """
    Save the DataFrame and run preprocessing via AnalysisPipeline.

    Args:
        args: Configuration (output_folder, preprocessing_folder, preprocessing_n_workers).
        df: DataFrame of ECG paths to preprocess.
        errors: Optional list to collect error messages; if None, errors are not collected.

    Raises:
        Any exception raised by AnalysisPipeline.save_and_preprocess_data.
    """
    AnalysisPipeline.save_and_preprocess_data(
        df=df,
        output_folder=args.output_folder,
        preprocessing_folder=args.preprocessing_folder,
        preprocessing_n_workers=args.preprocessing_n_workers,
        errors=errors,
    )

def perform_analysis(
    args: HearWiseArgs,
    df: pd.DataFrame,
    errors: list[str] | None = None,
) -> tuple[dict | None, pd.DataFrame | None]:
    """
    Run the analysis pipeline (model inference and metrics) on the given DataFrame.

    Args:
        args: Configuration (batch_size, devices, model names, API key path).
        df: DataFrame with diagnosis and ecg_path columns.
        errors: Optional list to collect error messages; if provided, failures append here and return (None, None).

    Returns:
        (metrics, df_probabilities) on success, or (None, None) if analysis failed and errors was provided.
    """
    hugging_face_api_key = read_api_key(args.hugging_face_api_key_path)['HUGGING_FACE_API_KEY']
    return AnalysisPipeline.run_analysis(
        df=df,
        batch_size=args.batch_size,
        diagnosis_classifier_device=args.diagnosis_classifier_device,
        signal_processing_device=args.signal_processing_device,
        signal_processing_model_name=args.signal_processing_model_name,
        diagnosis_classifier_model_name=args.diagnosis_classifier_model_name,
        hugging_face_api_key=hugging_face_api_key,
        errors=errors,
    )

def validate_dataframe(df: pd.DataFrame, diagnosis_to_file_columns: dict) -> tuple[list[str], list[str]]:
    """
    Ensure DataFrame has consistent diagnosis and file-name column pairs per the mapping.

    Args:
        df: Input DataFrame.
        diagnosis_to_file_columns: Map from diagnosis column name to ECG file name column name.

    Returns:
        (existing_diagnosis_columns, existing_file_columns) for columns present in df.

    Raises:
        ValueError: If any diagnosis column exists without its file column, or vice versa.
    """
    # Invert the mapping for reverse lookup
    file_to_diagnosis_columns = {v: k for k, v in diagnosis_to_file_columns.items()}
    
    # Sets for faster lookup
    diagnosis_columns_set = set(diagnosis_to_file_columns.keys())
    file_columns_set = set(diagnosis_to_file_columns.values())

    # Identify existing columns in the DataFrame
    existing_diagnosis_columns = diagnosis_columns_set.intersection(df.columns)
    existing_file_columns = file_columns_set.intersection(df.columns)
    
    missing_file_columns = []
    for diagnosis_column in existing_diagnosis_columns:
        expected_file_column = diagnosis_to_file_columns[diagnosis_column]
        if expected_file_column not in existing_file_columns:
            missing_file_columns.append(expected_file_column)
    
    missing_diagnosis_columns = []
    for file_column in existing_file_columns:
        expected_diagnosis_column = file_to_diagnosis_columns[file_column]
        if expected_diagnosis_column not in existing_diagnosis_columns:
            missing_diagnosis_columns.append(expected_diagnosis_column)
                
    error_messages = []
    if missing_file_columns:
        error_messages.append(
            f"Missing ECG file name columns corresponding to existing diagnosis columns: {missing_file_columns}"
        )
    if missing_diagnosis_columns:
        error_messages.append(
            f"Missing diagnosis columns corresponding to existing ECG file name columns: {missing_diagnosis_columns}"
        )
    
    # Raise error if any validation rules are violated
    if error_messages:
        full_error_message = "\n".join(error_messages)
        raise ValueError(f"DataFrame validation failed:\n{full_error_message}")
        
    return list(existing_diagnosis_columns), list(existing_file_columns)

def create_preprocessing_dataframe(df: pd.DataFrame, existing_file_columns: list[str], ecg_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a preprocessing DataFrame of ECG file paths and a DataFrame of missing files.

    Unpivots file columns into a single ecg_path column, filters by file existence,
    and deduplicates.

    Args:
        df: Source DataFrame.
        existing_file_columns: Column names containing ECG file names.
        ecg_path: Base directory to join with file names.

    Returns:
        (df_preprocessing, df_missing) where df_preprocessing has ecg_path and df_missing lists missing paths.
    """
    # Unpivot the DataFrame to have a single column of file paths
    melted_df = df.melt(value_vars=existing_file_columns, value_name='file_name').dropna(subset=['file_name'])
    # Construct full paths
    melted_df['ecg_path'] = melted_df['file_name'].apply(lambda x: os.path.join(ecg_path, x))
    
    # Create separate dataframes for existing and missing files
    exists_mask = melted_df['ecg_path'].apply(lambda x: os.path.exists(x))
    df_preprocessing = melted_df[exists_mask].reset_index(drop=True)
    df_missing = melted_df[~exists_mask].reset_index(drop=True)
    
    logger.info("Rows removed (files not found): %d", len(df_missing))
        
    # Remove duplicates from preprocessing df
    df_preprocessing = df_preprocessing[['ecg_path']].drop_duplicates().reset_index(drop=True) 
            
    return df_preprocessing, df_missing

def create_analysis_dataframe(df: pd.DataFrame, diagnosis_column: str, ecg_file_column: str, preprocessing_folder: str) -> pd.DataFrame:
    """
    Build an analysis DataFrame with diagnosis and paths to preprocessed .base64 ECG files.

    Drops rows with null diagnosis or file name, resolves paths under preprocessing_folder,
    and keeps only rows whose preprocessed file exists.

    Args:
        df: Source DataFrame.
        diagnosis_column: Column name for diagnosis labels.
        ecg_file_column: Column name for ECG file names.
        preprocessing_folder: Directory where preprocessed .base64 files are stored.

    Returns:
        DataFrame with columns diagnosis and ecg_path.
    """
    df_non_null = df[[diagnosis_column, ecg_file_column]].dropna(subset=[diagnosis_column, ecg_file_column])
    
    df_analysis = pd.DataFrame(
        {
            'diagnosis': df_non_null[diagnosis_column].tolist(),
            'ecg_path': [os.path.splitext(os.path.join(preprocessing_folder, os.path.basename(x)))[0] + ".base64" for x in df_non_null[ecg_file_column]]
            #'ecg_path': [os.path.splitext(os.path.join(preprocessing_folder, x))[0] + ".base64" for x in df_non_null[ecg_file_column]]
        }
    )
        
    # Remove files that do not exist
    df_analysis = df_analysis[df_analysis['ecg_path'].apply(os.path.exists)]
    # reset index to 0
    df_analysis = df_analysis.reset_index(drop=True)
    
    logger.info("Rows removed (files not found): %d", len(df) - len(df_analysis))
        
    return df_analysis

def main(args: HearWiseArgs):
    """
    Run the pipeline: load data, optionally preprocess and/or run analysis, and print any collected errors.

    Mode controls whether preprocessing, analysis, or both run. Errors are collected
    and printed at the end under "Errors encountered:".
    """
    if args.mode not in {Mode.PREPROCESSING, Mode.ANALYSIS, Mode.FULL_RUN}:
        raise ValueError(f"Invalid mode: {args.mode}. Please choose from 'preprocessing', 'analysis', or 'full_run'.")

    errors: list[str] = []

    try:
        logger.info("Loading input DataFrame from %s", args.data_path)
        df = load_df(args.data_path)
        logger.info("DataFrame loaded: %d rows", len(df))

        existing_diagnosis_columns, existing_file_columns = validate_dataframe(
            df=df,
            diagnosis_to_file_columns=DIAGNOSIS_TO_FILE_COLUMNS
        )
        set_up_directories(args)

        if args.mode == Mode.PREPROCESSING or args.mode == Mode.FULL_RUN:
            logger.info("Preprocessing data...")
            logger.info("Creating preprocessing dataframe...")
            df_preprocessing, df_missing = create_preprocessing_dataframe(
                df=df,
                existing_file_columns=existing_file_columns,
                ecg_path=args.ecg_signals_path
            )
            logger.info("Preprocessing dataframe created.")
            if not df_missing.empty:
                current_date = datetime.now().strftime('%Y%m%d')
                missing_files_path = os.path.join(args.output_folder, f'missing_files_{current_date}.csv')
                save_df(df_missing, missing_files_path)
                logger.info("Missing files list saved to %s", missing_files_path)
            logger.info("Saving and performing preprocessing...")
            save_and_perform_preprocessing(args, df_preprocessing, errors=errors)
            logger.info("Data preprocessed.")

        if args.mode == Mode.ANALYSIS or args.mode == Mode.FULL_RUN:
            for diagnosis_column in existing_diagnosis_columns:
                ecg_file_column = DIAGNOSIS_TO_FILE_COLUMNS[diagnosis_column]
                logger.info("Creating analysis dataframe for %s...", diagnosis_column)
                df_analysis = create_analysis_dataframe(
                    df=df,
                    diagnosis_column=diagnosis_column,
                    ecg_file_column=ecg_file_column,
                    preprocessing_folder=args.preprocessing_folder
                )
                logger.info("Analysis dataframe created for %s.", diagnosis_column)
                args.diagnosis_classifier_model_name = MODEL_MAPPING[diagnosis_column]['bert']
                signal_processing_models = []
                if args.use_wcr:
                    signal_processing_models.append(MODEL_MAPPING[diagnosis_column]['wcr'])
                if args.use_efficientnet:
                    signal_processing_models.append(MODEL_MAPPING[diagnosis_column]['efficientnet'])

                for model_name in signal_processing_models:
                    logger.info("Performing analysis with %s...", model_name)
                    args.signal_processing_model_name = model_name
                    metrics, df_probabilities = perform_analysis(args=args, df=df_analysis, errors=errors)

                    if metrics is None or df_probabilities is None:
                        continue

                    current_date = datetime.now().strftime('%Y%m%d')
                    logger.info("Saving metrics and probabilities...")
                    save_df(
                        df_probabilities,
                        os.path.join(
                            args.output_folder,
                            f'{model_name}_{current_date}_{diagnosis_column}_probabilities.csv'
                        )
                    )
                    save_json(
                        metrics,
                        os.path.join(
                            args.output_folder,
                            f'{model_name}_{current_date}_{diagnosis_column}.json'
                        )
                    )
                    save_to_csv(
                        metrics,
                        os.path.join(args.output_folder, f'{model_name}_{current_date}_{diagnosis_column}.csv')
                    )
                    logger.info("Metrics and probabilities saved.")
    except Exception as e:
        errors.append(f"[main] Fatal: {e}")

    if errors:
        print("Errors encountered:")
        for msg in errors:
            print(f"  {msg}")
        sys.exit(1)

if __name__ == "__main__":
    args = HearWiseArgs.parse_arguments()
    logger.info(
        "Run config: mode=%s data_path=%s output_folder=%s batch_size=%s "
        "diagnosis_device=%s signal_device=%s use_wcr=%s use_efficientnet=%s",
        args.mode,
        args.data_path,
        args.output_folder,
        args.batch_size,
        args.diagnosis_classifier_device,
        args.signal_processing_device,
        args.use_wcr,
        args.use_efficientnet,
    )
    main(args)

