import os
import pandas as pd

from utils.constants import Mode
from utils.parser import HearWiseArgs
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import save_to_csv, save_json, read_api_key, load_df, set_path, save_df


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

def load_and_prepare_data(args: HearWiseArgs, new_path: str, new_ext: str = None) -> pd.DataFrame:
    """
    Load data from a specified path, preprocess it, and prepare it for analysis.

    This function performs several data loading and preprocessing steps:
    - Loads the DataFrame from the provided `data_path`.
    - Removes rows where the 'diagnosis' column is empty, reporting the number of removed rows.
    - Sets the path for ECG signals using the `new_path`.
    - Optionally changes the file extension of ECG paths if `new_ext` is provided.
    - Validates that the DataFrame is not empty after preprocessing.
    - Ensures that the 'ecg_path' column exists and that all referenced files exist.

    Args:
        args (HearWiseArgs): Configuration arguments containing `data_path` and other settings.
        new_path (str): The new directory path to set for ECG signal files.
        new_ext (str, optional): New file extension for ECG paths. If provided, updates the 
                                 'ecg_path' column with this new extension. Defaults to None.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for further analysis.

    Raises:
        ValueError: If the resulting DataFrame is empty or if the 'ecg_path' column is missing.
        FileNotFoundError: If any of the files specified in 'ecg_path' do not exist.
    """
    # Read data
    df = load_df(args.data_path)
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. Please provide a valid data file.")    
    
    # Check if 'ecg_file_name' column exists and if the files exist
    if 'ecg_file_name' not in df.columns:
        raise ValueError(f"'ecg_file_name' column is missing in the DataFrame.")
    
    # Check if 'diagnosis' column exists
    if 'diagnosis' not in df.columns:
        raise ValueError("'diagnosis' column is missing in the DataFrame.")    
    
    # Remove rows with empty 'diagnosis' column and count them
    missing_diagnosis_count = df['diagnosis'].isna().sum()
    df = df.dropna(subset=['diagnosis']).reset_index(drop=True)
    if missing_diagnosis_count > 0:
        print(f"Removed {missing_diagnosis_count} rows with empty 'diagnosis' column.")
    
    # Set path to ecg signals
    df = set_path(df, new_path)

    # Change extension of ecg_path if ext is not None
    if new_ext is not None:
        df['ecg_path'] = df['ecg_path'].apply(lambda x: os.path.splitext(x)[0] + new_ext)    

    # Check if the files in 'ecg_path' column exist
    missing_files = df[~df['ecg_path'].apply(os.path.exists)]
    if not missing_files.empty:
        missing_files_list = missing_files['ecg_path'].tolist()
        missing_files_df = pd.DataFrame(missing_files_list, columns=['missing_files'])
        missing_path = os.path.join(args.output_folder, 'missing_files.csv')
        missing_files_df.to_csv(missing_path, index=False)
        print(f'Warning: Missing files saved to {missing_path} - List of missing files:')
        print(missing_files_df)
        
        # Discard missing_files from df
        df = df[df['ecg_path'].apply(os.path.exists)].reset_index(drop=True)
        print(f'Warning: {len(missing_files)} files were missing and discarded from the DataFrame.')
        
    return df

def save_and_perform_preprocessing(args: HearWiseArgs, df: pd.DataFrame):
    """
    Save the DataFrame and perform preprocessing using the AnalysisPipeline.

    This function delegates the task of saving and preprocessing data to the 
    `AnalysisPipeline`. It utilizes the provided arguments to determine the 
    output and preprocessing directories, as well as the number of worker threads 
    for preprocessing.

    Args:
        args (HearWiseArgs): Configuration arguments containing paths for output and 
                             preprocessing folders, and the number of preprocessing workers.
        df (pd.DataFrame): The DataFrame containing the data to preprocess.

    Raises:
        Any exceptions raised by `AnalysisPipeline.save_and_preprocess_data`.
    """     
    AnalysisPipeline.save_and_preprocess_data(
        df=df, 
        output_folder=args.output_folder,
        preprocessing_folder=args.preprocessing_folder,
        preprocessing_n_workers=args.preprocessing_n_workers
    )

def perform_analysis(args: HearWiseArgs, df: pd.DataFrame):
    """
    Perform analysis on the prepared DataFrame using the AnalysisPipeline.

    This function reads the Hugging Face API key and runs the analysis pipeline 
    with the specified configurations such as batch size, devices, and model names.

    Args:
        args (HearWiseArgs): Configuration arguments containing paths for API keys, 
                             batch size, device specifications, and model names.
        df (pd.DataFrame): The DataFrame containing the data to analyze.

    Returns:
        tuple: A tuple containing `metrics` and `df_probabilities` resulting from the analysis.

    Raises:
        Any exceptions raised by `read_api_key` or `AnalysisPipeline.run_analysis`.
    """    
    hugging_face_api_key = read_api_key(args.hugging_face_api_key_path)['HUGGING_FACE_API_KEY']
    return AnalysisPipeline.run_analysis(
        df=df,
        batch_size=args.batch_size,
        diagnosis_classifier_device=args.diagnosis_classifier_device,
        signal_processing_device=args.signal_processing_device,
        signal_processing_model_name=args.signal_processing_model_name,
        diagnosis_classifier_model_name=args.diagnosis_classifier_model_name,
        hugging_face_api_key=hugging_face_api_key
    )

def main(args: HearWiseArgs):
    if args.mode not in {Mode.PREPROCESSING, Mode.ANALYSIS, Mode.FULL_RUN}:
        raise ValueError(f"Invalid mode: {args.mode}. Please choose from 'preprocessing', 'analysis', or 'full_run'.")
    
    # Set up directories
    set_up_directories(args)

    if args.mode in {Mode.PREPROCESSING, Mode.FULL_RUN}:
        # Load and prepare data
        df = load_and_prepare_data(args, new_path=args.ecg_signals_path)
        
        # Preprocess data
        save_and_perform_preprocessing(args, df)
        
    if args.mode in {Mode.ANALYSIS, Mode.FULL_RUN}:        
        df = load_and_prepare_data(args, new_path=args.preprocessing_folder, new_ext=".base64")
        
        metrics, df_probabilities = perform_analysis(args, df)

        # Save metrics and probabilities
        from datetime import datetime
        current_date = datetime.now().strftime('%Y%m%d')
        model_name = args.signal_processing_model_name

        save_df(df_probabilities, os.path.join(args.output_folder, f'{args.output_file}_{current_date}_{model_name}_probabilities.csv'))
        save_json(metrics, os.path.join(args.output_folder, f'{args.output_file}_{current_date}_{model_name}.json'))
        save_to_csv(metrics, os.path.join(args.output_folder, f'{args.output_file}_{current_date}_{model_name}.csv'))

if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    main(args)

