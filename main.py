import os
import pandas as pd

from utils.constants import Mode
from utils.parser import HearWiseArgs
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import save_to_csv, save_json, read_api_key, save_df, load_and_prepare_data


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
        print(f"Loading and preparing data from {args.ecg_signals_path}...")
        df = load_and_prepare_data(args, new_path=args.ecg_signals_path)
        print(f"Data loaded and prepared.")

        # Preprocess data
        print(f"Preprocessing data...")
        save_and_perform_preprocessing(args, df)
        print(f"Data preprocessed.")
        
    if args.mode in {Mode.ANALYSIS, Mode.FULL_RUN}:        
        print(f"Loading and preparing data from {args.preprocessing_folder}...")
        df = load_and_prepare_data(args, new_path=args.preprocessing_folder, new_ext=".base64")
        print(f"Data loaded and prepared.")
        
        print(f"Performing analysis...")
        metrics, df_probabilities = perform_analysis(args, df)
        print(f"Analysis performed.")

        # Save metrics and probabilities
        from datetime import datetime
        current_date = datetime.now().strftime('%Y%m%d')
        model_name = args.signal_processing_model_name

        print(f"Saving metrics and probabilities...")
        save_df(df_probabilities, os.path.join(args.output_folder, f'{args.output_file}_{current_date}_{model_name}_probabilities.csv'))
        save_json(metrics, os.path.join(args.output_folder, f'{args.output_file}_{current_date}_{model_name}.json'))
        save_to_csv(metrics, os.path.join(args.output_folder, f'{args.output_file}_{current_date}_{model_name}.csv'))
        print(f"Metrics and probabilities saved.")

if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    main(args)

