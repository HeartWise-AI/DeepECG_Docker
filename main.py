import os
import torch
import argparse

from utils.parser import HearWiseArgs
from models import HeartWiseModelFactory
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import save_to_csv, save_to_json, read_api_key


def main(args: HearWiseArgs):
    data_path = args.data_path
    batch_size = args.batch_size
    output_file = args.output_file
    output_folder = args.output_folder
    diagnosis_classifier_device = torch.device(args.diagnosis_classifier_device)
    signal_processing_device = torch.device(args.signal_processing_device)
    huggingface_api_key_path = args.huggingface_api_key_path
    signal_processing_model_name = args.signal_processing_model_name
    diagnosis_classifier_model_name = args.diagnosis_classifier_model_name
    
    huggingface_api_key = read_api_key(huggingface_api_key_path)['HUGGINGFACE_API_KEY']
    
    diagnosis_classifier_model = HeartWiseModelFactory.create_model(
        {
            'model_name': diagnosis_classifier_model_name,
            'map_location': diagnosis_classifier_device, 
            'huggingface_api_key': huggingface_api_key
        }
    )
        
    signal_processing_model = HeartWiseModelFactory.create_model(
        {
            'model_name': signal_processing_model_name,
            'map_location': signal_processing_device,
            'huggingface_api_key': huggingface_api_key
        }
    )
    
    metrics = AnalysisPipeline.run_analysis(
        data_path=data_path,
        batch_size=batch_size,
        signal_processing_model=signal_processing_model,
        diagnosis_classifier_model=diagnosis_classifier_model
    )

    save_to_json(metrics, os.path.join(output_folder, f'{output_file}.json'))        
    save_to_csv(metrics, os.path.join(output_folder, f'{output_file}.csv'))

        
if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    main(args)

