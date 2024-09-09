import os
import pandas as pd

from utils.parser import HearWiseArgs
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import save_to_csv, save_to_json, read_api_key


def main(args: HearWiseArgs):
    df = pd.read_csv(args.data_path)

    AnalysisPipeline.preprocess_data(
        df=df, 
        output_folder=args.output_folder,
    )

    hugging_face_api_key = read_api_key(args.hugging_face_api_key_path)['HUGGING_FACE_API_KEY']
        
    metrics = AnalysisPipeline.run_analysis(
        df=df,
        batch_size=args.batch_size,
        diagnosis_classifier_device=args.diagnosis_classifier_device,
        signal_processing_device=args.signal_processing_device,
        signal_processing_model_name=args.signal_processing_model_name,
        diagnosis_classifier_model_name=args.diagnosis_classifier_model_name,
        hugging_face_api_key=hugging_face_api_key
    )

    output_folder = args.output_folder
    save_to_json(metrics, os.path.join(output_folder, f'{args.output_file}.json'))        
    save_to_csv(metrics, os.path.join(output_folder, f'{args.output_file}.csv'))

        
if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    main(args)

