import os
import sys
import warnings
import pandas as pd
from datetime import datetime
from PIL import Image

warnings.filterwarnings("ignore", message=".*_register_pytree_node.*deprecated", category=FutureWarning)

from utils.constants import (
    Mode, 
    DIAGNOSIS_TO_FILE_COLUMNS,
    MODEL_MAPPING,
    PNG_PATH_COLUMN,
    IMAGE_EXTENSIONS,
    UNSUPPORTED_IMAGE_EXTENSIONS,
    DEEPECG_PNG_OUTPUTS,
    ECG_PATTERNS
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

def save_and_perform_preprocessing(args: HearWiseArgs, df: pd.DataFrame, errors: list[str] | None = None) -> pd.DataFrame | None:
    """
    Save the DataFrame and run preprocessing via AnalysisPipeline.

    Args:
        args: Configuration (output_folder, preprocessing_folder, preprocessing_n_workers).
        df: DataFrame of ECG paths to preprocess.
        errors: Optional list to collect error messages; if None, errors are not collected.

    Returns:
        Preprocessed DataFrame, or None if preprocessing failed completely.
    """
    return AnalysisPipeline.save_and_preprocess_data(
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
    # Images (DeepECG-PNG) are read as they are: no signal preprocessing
    lower_paths = melted_df['ecg_path'].astype(str).str.lower()
    image_mask = lower_paths.str.endswith(IMAGE_EXTENSIONS)
    unsupported_mask = lower_paths.str.endswith(UNSUPPORTED_IMAGE_EXTENSIONS)
    df_preprocessing = melted_df[exists_mask & ~image_mask & ~unsupported_mask].reset_index(drop=True)
    df_missing = melted_df[~exists_mask].reset_index(drop=True)
    
    logger.info("Rows removed (files not found): %d", len(df_missing))
    logger.info("Image files (not preprocessed): %d", int((exists_mask & image_mask).sum()))
    if unsupported_mask.any():
        logger.warning("Unsupported image files skipped (only %s): %d", ", ".join(IMAGE_EXTENSIONS),
                       melted_df.loc[unsupported_mask, 'ecg_path'].nunique())
        
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

def expand_png_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a CSV with one PNG per row into the per-task layout the pipeline uses.

    When the DataFrame has a `png_path` column, that path becomes the file column of every
    task (unless the CSV already has it), and missing label columns are added empty, so each
    PNG runs through every task; labels stay optional.

    Args:
        df: Source DataFrame.

    Returns:
        The DataFrame with per-task file and diagnosis columns.
    """
    if PNG_PATH_COLUMN not in df.columns:
        return df
    df = df.copy()
    for diagnosis_column, file_column in DIAGNOSIS_TO_FILE_COLUMNS.items():
        if file_column not in df.columns:
            df[file_column] = df[PNG_PATH_COLUMN]
        if diagnosis_column not in df.columns:
            df[diagnosis_column] = pd.Series([None] * len(df), index=df.index, dtype=object)
    logger.info("Using '%s' (%d rows) for every task", PNG_PATH_COLUMN, len(df))
    return df

def create_image_analysis_dataframe(df: pd.DataFrame, diagnosis_column: str, ecg_file_column: str, ecg_path: str) -> pd.DataFrame:
    """
    Build an analysis DataFrame for an image model: one row per image file of the task.

    Rows without a label are kept (inference only); rows without an image file, with a
    non-image file or with a file that does not exist are dropped.

    Args:
        df: Source DataFrame.
        diagnosis_column: Column name for diagnosis labels.
        ecg_file_column: Column name for the image file names.
        ecg_path: Base directory to join with file names (absolute paths are kept).

    Returns:
        DataFrame with columns diagnosis, ecg_path and file_name (as given in the CSV).
    """
    df_non_null = df[[diagnosis_column, ecg_file_column]].dropna(subset=[ecg_file_column])
    names = df_non_null[ecg_file_column].astype(str)
    is_image = names.str.lower().str.endswith(IMAGE_EXTENSIONS)
    labels = df_non_null[diagnosis_column][is_image]
    if diagnosis_column != 'ecg_machine_diagnosis':
        labels = normalize_binary_labels(labels, diagnosis_column)
    df_analysis = pd.DataFrame(
        {
            'diagnosis': labels.tolist(),
            'ecg_path': [os.path.join(ecg_path, x) for x in names[is_image]],
            'file_name': names[is_image].tolist()
        },
        columns=['diagnosis', 'ecg_path', 'file_name']
    )
    if df_analysis.empty:
        logger.info("No image rows for %s.", diagnosis_column)
        return df_analysis
    exists = df_analysis['ecg_path'].map(os.path.exists).astype(bool)
    if (~exists).any():
        logger.warning("Images not found for %s: %d", diagnosis_column, int((~exists).sum()))
    df_analysis = df_analysis[exists].reset_index(drop=True)
    readable = df_analysis['ecg_path'].map(lambda p: image_error(p) is None).astype(bool)
    if (~readable).any():
        logger.warning("Unreadable images skipped for %s: %d", diagnosis_column, int((~readable).sum()))
    df_analysis = df_analysis[readable].reset_index(drop=True)
    if diagnosis_column == 'ecg_machine_diagnosis':
        n_labelled = int(df_analysis['diagnosis'].map(lambda t: pd.notna(t) and bool(str(t).strip())).sum())
    else:
        n_labelled = int(df_analysis['diagnosis'].notna().sum())
    logger.info("Image rows for %s: %d (%d with a label)", diagnosis_column, len(df_analysis), n_labelled)
    return df_analysis

# Images that could not be decoded, path -> error (checked once per file per run)
UNREADABLE_IMAGES: dict[str, str | None] = {}

def image_error(path: str) -> str | None:
    """Decode the image once; return None if it reads, else the error (cached per path)."""
    if path not in UNREADABLE_IMAGES:
        try:
            with Image.open(path) as image:
                image.load()
            UNREADABLE_IMAGES[path] = None
        except Exception as e:
            UNREADABLE_IMAGES[path] = f"{type(e).__name__}: {e}"
    return UNREADABLE_IMAGES[path]

def normalize_binary_labels(labels: pd.Series, task: str) -> pd.Series:
    """
    0/1 labels from numbers, numeric strings or booleans (true/false, yes/no). Any other
    value is treated as missing (the row still gets a prediction) and counted in a warning.
    """
    words = {'true': 1, 'false': 0, 'yes': 1, 'no': 0}
    mapped = labels.map(lambda v: words.get(str(v).strip().lower(), v) if pd.notna(v) else v)
    numeric = pd.to_numeric(mapped, errors='coerce')
    invalid = labels.notna() & ~numeric.isin([0, 1])
    if invalid.any():
        logger.warning("Labels of %s that are not 0/1 treated as missing: %d", task, int(invalid.sum()))
        numeric[invalid] = float('nan')
    return numeric

def combine_deepecg_png_outputs(per_task: dict) -> pd.DataFrame:
    """
    One row per image with every DeepECG-PNG output: SHD, LVEF<=40, LVEF<50 and iAFib5
    probability and 0/1 prediction (validation threshold), then the 77 class probabilities.

    Args:
        per_task: diagnosis column -> the task's probabilities DataFrame.

    Returns:
        The combined DataFrame (file_name first).
    """
    order = ['shd', 'lvef_40', 'lvef_50', 'afib_5y', 'ecg_machine_diagnosis']
    tasks = [t for t in order if t in per_task]
    # one row per image, in input order (an outer merge would sort the file names)
    file_names = pd.unique(pd.concat([per_task[t]['file_name'] for t in tasks], ignore_index=True))
    combined = pd.DataFrame({'file_name': file_names})
    for diagnosis_column in tasks:
        task_df = per_task[diagnosis_column].drop_duplicates(subset='file_name')
        if len(task_df) < len(per_task[diagnosis_column]):
            logger.warning("%s: %d duplicate image rows kept once in the combined file", diagnosis_column,
                           len(per_task[diagnosis_column]) - len(task_df))
        if diagnosis_column == 'ecg_machine_diagnosis':
            part = task_df[['file_name'] + [f"{p}_sig_model" for p in ECG_PATTERNS]].rename(
                columns={f"{p}_sig_model": f"{p}_prob" for p in ECG_PATTERNS})
        else:
            output = DEEPECG_PNG_OUTPUTS[diagnosis_column]
            part = task_df[['file_name', 'predictions', 'prediction']].rename(
                columns={'predictions': f"{output}_prob", 'prediction': f"{output}_pred"})
        combined = combined.merge(part, on='file_name', how='left')
    return combined

def main(args: HearWiseArgs):
    """
    Run the pipeline: load data, optionally preprocess and/or run analysis, and print any collected errors.

    Mode controls whether preprocessing, analysis, or both run. Errors are collected
    and printed at the end under "Errors encountered:".
    """
    if args.mode not in {Mode.PREPROCESSING, Mode.ANALYSIS, Mode.FULL_RUN}:
        raise ValueError(f"Invalid mode: {args.mode}. Please choose from 'preprocessing', 'analysis', or 'full_run'.")

    errors: list[str] = []
    UNREADABLE_IMAGES.clear()

    try:
        logger.info("Loading input DataFrame from %s", args.data_path)
        df = load_df(args.data_path)
        logger.info("DataFrame loaded: %d rows", len(df))
        df = expand_png_column(df)
        # Image-only tasks (shd) are expected only when the image model is on, so a signal CSV
        # with an unrelated `shd` column validates as before.
        task_columns = {
            diagnosis_column: file_column
            for diagnosis_column, file_column in DIAGNOSIS_TO_FILE_COLUMNS.items()
            if getattr(args, 'use_deepecg_png', False)
            or MODEL_MAPPING[diagnosis_column]['wcr'] or MODEL_MAPPING[diagnosis_column]['efficientnet']
        }

        existing_diagnosis_columns, existing_file_columns = validate_dataframe(
            df=df,
            diagnosis_to_file_columns=task_columns
        )
        file_references = df[existing_file_columns].stack().astype(str).str.lower() if existing_file_columns else pd.Series(dtype=str)
        has_images = bool(file_references.str.endswith(IMAGE_EXTENSIONS).any())
        has_signals = bool((~file_references.str.endswith(IMAGE_EXTENSIONS + UNSUPPORTED_IMAGE_EXTENSIONS)).any())
        if has_images and not getattr(args, 'use_deepecg_png', False):
            logger.warning("The CSV lists PNG images but use_deepecg_png is False: images are not analysed. "
                           "Set use_deepecg_png: True in heartwise.config.")
        set_up_directories(args)

        preprocessing_failed = False
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
                current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
                missing_files_path = os.path.join(args.output_folder, f'missing_files_{current_date}.csv')
                save_df(df_missing, missing_files_path)
                logger.info("Missing files list saved to %s", missing_files_path)
            if not has_signals:
                logger.info("No signal files to preprocess.")
            else:
                logger.info("Saving and performing preprocessing...")
                preprocessed_df = save_and_perform_preprocessing(args, df_preprocessing, errors=errors)
                if preprocessed_df is None:
                    logger.warning("Preprocessing failed completely.")
                    preprocessing_failed = True
                else:
                    logger.info("Data preprocessed.")

        if args.mode == Mode.ANALYSIS or args.mode == Mode.FULL_RUN:
            deepecg_png_outputs = {}
            for diagnosis_column in existing_diagnosis_columns:
                ecg_file_column = task_columns[diagnosis_column]
                args.diagnosis_classifier_model_name = MODEL_MAPPING[diagnosis_column]['bert']
                models_to_run = []   # (model name, analysis dataframe)
                signal_processing_models = []
                # signal models need the preprocessed signals; image models do not
                if not preprocessing_failed:
                    if args.use_wcr and MODEL_MAPPING[diagnosis_column]['wcr']:
                        signal_processing_models.append(MODEL_MAPPING[diagnosis_column]['wcr'])
                    if args.use_efficientnet and MODEL_MAPPING[diagnosis_column]['efficientnet']:
                        signal_processing_models.append(MODEL_MAPPING[diagnosis_column]['efficientnet'])
                if signal_processing_models:
                    logger.info("Creating analysis dataframe for %s...", diagnosis_column)
                    df_analysis = create_analysis_dataframe(
                        df=df,
                        diagnosis_column=diagnosis_column,
                        ecg_file_column=ecg_file_column,
                        preprocessing_folder=args.preprocessing_folder
                    )
                    logger.info("Analysis dataframe created for %s.", diagnosis_column)
                    models_to_run += [(model_name, df_analysis) for model_name in signal_processing_models]
                if getattr(args, 'use_deepecg_png', False) and MODEL_MAPPING[diagnosis_column]['deepecg_png']:
                    df_image_analysis = create_image_analysis_dataframe(
                        df=df,
                        diagnosis_column=diagnosis_column,
                        ecg_file_column=ecg_file_column,
                        ecg_path=args.ecg_signals_path
                    )
                    models_to_run.append((MODEL_MAPPING[diagnosis_column]['deepecg_png'], df_image_analysis))

                for model_name, df_model in models_to_run:
                    # (signal models keep their earlier behaviour on an empty frame)
                    if df_model.empty and model_name == MODEL_MAPPING[diagnosis_column]['deepecg_png']:
                        logger.warning("No image for %s; %s skipped.", diagnosis_column, model_name)
                        continue
                    logger.info("Performing analysis with %s...", model_name)
                    args.signal_processing_model_name = model_name
                    metrics, df_probabilities = perform_analysis(args=args, df=df_model, errors=errors)

                    if metrics is None or df_probabilities is None:
                        continue
                    if model_name == MODEL_MAPPING[diagnosis_column]['deepecg_png']:
                        deepecg_png_outputs[diagnosis_column] = df_probabilities

                    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
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

            unreadable = {p: e for p, e in UNREADABLE_IMAGES.items() if e is not None}
            if unreadable:
                current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
                unreadable_path = os.path.join(args.output_folder, f'unreadable_images_{current_date}.csv')
                save_df(pd.DataFrame({'ecg_path': list(unreadable), 'error': list(unreadable.values())}),
                        unreadable_path)
                logger.warning("%d unreadable images listed in %s", len(unreadable), unreadable_path)
            if deepecg_png_outputs:
                current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
                combined_path = os.path.join(args.output_folder, f'deepecg_png_{current_date}_all_tasks.csv')
                save_df(combine_deepecg_png_outputs(deepecg_png_outputs), combined_path)
                logger.info("DeepECG-PNG outputs (one row per image) saved to %s", combined_path)
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
        "diagnosis_device=%s signal_device=%s use_wcr=%s use_efficientnet=%s use_deepecg_png=%s",
        args.mode,
        args.data_path,
        args.output_folder,
        args.batch_size,
        args.diagnosis_classifier_device,
        args.signal_processing_device,
        args.use_wcr,
        args.use_efficientnet,
        args.use_deepecg_png,
    )
    main(args)

