import os
import pandas as pd

from utils.constants import Mode
from utils.parser import HearWiseArgs
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import save_to_csv, save_json, read_api_key, load_df, set_path, save_df


def set_up_directories(args: HearWiseArgs):
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    # Create preprocessing folder
    os.makedirs(args.preprocessing_folder, exist_ok=True)

def load_and_prepare_data(args: HearWiseArgs) -> pd.DataFrame:
    # Read data
    df = load_df(args.data_path)
    df = set_path(df, args.ecg_signals_path)
    return df

def perform_preprocessing(args: HearWiseArgs, df: pd.DataFrame):
    return AnalysisPipeline.preprocess_data(
        df=df, 
        output_folder=args.output_folder,
        preprocessing_folder=args.preprocessing_folder,
        preprocessing_n_workers=args.preprocessing_n_workers
    )

def perform_analysis(args: HearWiseArgs, df: pd.DataFrame):
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

    # Read data
    df = load_and_prepare_data(args)

    if args.mode in {Mode.PREPROCESSING, Mode.FULL_RUN}:
        # Preprocess data
        df_cleaned_ecg_signals = perform_preprocessing(args, df)
        save_df(df_cleaned_ecg_signals, os.path.join(args.preprocessing_folder, f"{args.output_file}_cleaned_ecg_signals.csv"))
        
    if args.mode in {Mode.ANALYSIS, Mode.FULL_RUN}:
        if args.mode == Mode.ANALYSIS:
            df = load_df(os.path.join(args.preprocessing_folder, f"{args.output_file}_cleaned_ecg_signals.csv"))
            
        metrics, df_probabilities = perform_analysis(args, df)

        # Save metrics and probabilities
        save_df(df_probabilities, os.path.join(args.output_folder, f'{args.output_file}_probabilities.csv'))
        save_json(metrics, os.path.join(args.output_folder, f'{args.output_file}.json'))
        save_to_csv(metrics, os.path.join(args.output_folder, f'{args.output_file}.csv'))

if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    main(args)

