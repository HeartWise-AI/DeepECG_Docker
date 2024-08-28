import os
import torch
from models import ModelFactory
from utils.analysis_pipeline import AnalysisPipeline, convert_to_df, save_to_csv, save_to_json



def main():
    output_folder = 'results'
    df_path = '../Llava-ECG/hw_QA_generator/parquet_files/npy_data_validatedByMD_test.parquet'
    device = torch.device('cuda:0')
    bert_classifier = ModelFactory.create_model(
        {
            'model_name': 'bert_diagnosis2classification',
            'map_location': device
        }
    )
        
    efficient_netV2 = ModelFactory.create_model(
        {
            'model_name': 'efficientnetv2',
            'map_location': device
        }
    )
    
    metrics = AnalysisPipeline.run_analysis(
        df_path=df_path,
        signal_processing_model=efficient_netV2,
        diagnosis_classifier_model=bert_classifier
    )

    save_to_json(metrics, os.path.join(output_folder, 'metrics.json'))        
    save_to_csv(metrics, os.path.join(output_folder, 'metrics.csv'))

        
if __name__ == '__main__':
    main()

