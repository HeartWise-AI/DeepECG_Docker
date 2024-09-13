import os
import pandas as pd

from utils.parser import HearWiseArgs
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import save_to_csv, save_to_json, read_api_key, load_csv_df, set_path
    
def main(args: HearWiseArgs):
    # Create tmp folder
    os.makedirs('./tmp', exist_ok=True)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Read data
    df = load_csv_df(args.data_path)
    df = set_path(df, args.ecg_signals_path)
    
    # Preprocess data
    df = AnalysisPipeline.preprocess_data(
        df=df, 
        output_folder=args.output_folder,
    )
    
    # Read API key
    hugging_face_api_key = read_api_key(args.hugging_face_api_key_path)['HUGGING_FACE_API_KEY']
        
    # Run analysis
    metrics, df_probabilities = AnalysisPipeline.run_analysis(
        df=df,
        batch_size=args.batch_size,
        diagnosis_classifier_device=args.diagnosis_classifier_device,
        signal_processing_device=args.signal_processing_device,
        signal_processing_model_name=args.signal_processing_model_name,
        diagnosis_classifier_model_name=args.diagnosis_classifier_model_name,
        hugging_face_api_key=hugging_face_api_key
    )

    # Save metrics and probabilities
    output_file = args.output_file
    output_folder = args.output_folder
    df_probabilities.to_csv(os.path.join(output_folder, f'{output_file}_probabilities.csv'), index=False)
    save_to_json(metrics, os.path.join(output_folder, f'{output_file}.json'))        
    save_to_csv(metrics, os.path.join(output_folder, f'{output_file}.csv'))
    
    # Remove tmp folder
    if os.path.exists('./tmp'):
        os.remove('./tmp')

if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    main(args)

