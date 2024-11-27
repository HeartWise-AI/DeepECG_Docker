import os
import pandas as pd

from datetime import datetime
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
    load_df
)

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

def perform_analysis(args: HearWiseArgs, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
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

def validate_dataframe(df: pd.DataFrame, diagnosis_to_file_columns: dict) -> tuple[list[str], list[str]]:

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

def create_preprocessing_dataframe(df: pd.DataFrame, existing_file_columns: list[str], ecg_path: str) -> pd.DataFrame:
    # Unpivot the DataFrame to have a single column of file paths
    melted_df = df.melt(value_vars=existing_file_columns, value_name='file_name').dropna(subset=['file_name'])
    # Construct full paths
    melted_df['ecg_path'] = melted_df['file_name'].apply(lambda x: os.path.join(ecg_path, x))
    
    # Remove files that do not exist
    df_preprocessing = melted_df[melted_df['ecg_path'].apply(os.path.exists)].reset_index(drop=True)
    
    print(f"Number of rows removed: {len(melted_df) - len(df_preprocessing)} because the files did not exist")
        
    # Remove duplicates
    df_preprocessing = df_preprocessing[['ecg_path']].drop_duplicates().reset_index(drop=True) 
            
    return df_preprocessing

def create_analysis_dataframe(df: pd.DataFrame, diagnosis_column: str, ecg_file_column: str, preprocessing_folder: str) -> pd.DataFrame:
    df_non_null = df[[diagnosis_column, ecg_file_column]].dropna(subset=[diagnosis_column, ecg_file_column])
    
    df_analysis = pd.DataFrame(
        {
            'diagnosis': df_non_null[diagnosis_column].tolist(),
            'ecg_path': [os.path.splitext(os.path.join(preprocessing_folder, x))[0] + ".base64" for x in df_non_null[ecg_file_column]]
        }
    )
        
    # Remove files that do not exist
    df_analysis = df_analysis[df_analysis['ecg_path'].apply(os.path.exists)]
    # reset index to 0
    df_analysis = df_analysis.reset_index(drop=True)
    
    print(f"Number of rows removed: {len(df) - len(df_analysis)} because the files did not exist")
        
    return df_analysis

def main(args: HearWiseArgs):
    if args.mode not in {Mode.PREPROCESSING, Mode.ANALYSIS, Mode.FULL_RUN}:
        raise ValueError(f"Invalid mode: {args.mode}. Please choose from 'preprocessing', 'analysis', or 'full_run'.")
        
    df = load_df(args.data_path)
        
    existing_diagnosis_columns, existing_file_columns = validate_dataframe(
        df=df, 
        diagnosis_to_file_columns=DIAGNOSIS_TO_FILE_COLUMNS
    )   
    
    # Set up directories
    set_up_directories(args)

    # Preprocess data
    if args.mode == Mode.PREPROCESSING or args.mode == Mode.FULL_RUN: 
        print(f"Preprocessing data...")

        # Create preprocessing dataframe
        print(f"Creating preprocessing dataframe...")
        df_preprocessing = create_preprocessing_dataframe(
            df=df, 
            existing_file_columns=existing_file_columns, 
            ecg_path=args.ecg_signals_path
        )
        print(f"Preprocessing dataframe created.")
        
        # Save and perform preprocessing
        print(f"Saving and performing preprocessing...")
        save_and_perform_preprocessing(args, df_preprocessing)
        print(f"Data preprocessed.")

    if args.mode == Mode.ANALYSIS or args.mode == Mode.FULL_RUN:       
        # Iterate over each diagnosis column
        for diagnosis_column in existing_diagnosis_columns:
            # Create a test dataframe with only the diagnosis column and the corresponding ECG file name column
            ecg_file_column = DIAGNOSIS_TO_FILE_COLUMNS[diagnosis_column]
            
            print(f"Creating analysis dataframe for {diagnosis_column}...")
            df_analysis = create_analysis_dataframe(
                df=df, 
                diagnosis_column=diagnosis_column, 
                ecg_file_column=ecg_file_column, 
                preprocessing_folder=args.preprocessing_folder
            )
            print(f"Analysis dataframe created for {diagnosis_column}.")
            
            # Append default bert model to the list of signal processing models to heartwise args
            args.diagnosis_classifier_model_name = MODEL_MAPPING[diagnosis_column]['bert']
            
            # Append signal processing models to the list of signal processing models to heartwise args
            signal_processing_models = []
            if args.use_wcr:
                signal_processing_models.append(
                    MODEL_MAPPING[diagnosis_column]['wcr']
                )
            if args.use_efficientnet:
                signal_processing_models.append(
                    MODEL_MAPPING[diagnosis_column]['efficientnet']
                )
                 
            # iterate over signal processing models
            for model_name in signal_processing_models:
                print(f"Performing analysis with {model_name}...")
                
                # Set signal processing model name
                args.signal_processing_model_name = model_name
                
                # Perform analysis
                metrics, df_probabilities = perform_analysis(
                    args=args, 
                    df=df_analysis
                )

                # Save metrics and probabilities
                current_date = datetime.now().strftime('%Y%m%d')

                print(f"Saving metrics and probabilities...")
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
                print(f"Metrics and probabilities saved.")

if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    print("Summary of arguments:")
    print(f"Diagnosis Classifier Device: {args.diagnosis_classifier_device}")
    print(f"Signal Processing Device: {args.signal_processing_device}")
    print(f"Data Path: {args.data_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output Folder: {args.output_folder}")
    print(f"Hugging Face API Key Path: {args.hugging_face_api_key_path}")
    print(f"Use WCR: {args.use_wcr}")
    print(f"Use EfficientNet: {args.use_efficientnet}")
    print(f"ECG Signals Path: {args.ecg_signals_path}")
    print(f"Mode: {args.mode}")
    print(f"Preprocessing Folder: {args.preprocessing_folder}")
    print(f"Preprocessing Number of Workers: {args.preprocessing_n_workers}")
    main(args)

