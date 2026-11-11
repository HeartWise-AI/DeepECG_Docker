#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: source run_pipeline.bash --mode <mode> --csv_file <file.csv>"
    echo "  --mode       Specify the mode of operation (e.g., analysis)"
    echo "  --csv_file   Specify the path to the CSV file"
    return 1
}

# Function to get value of a parameter from config file
get_param() {
    grep "^$1:" heartwise.config | cut -d':' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Read parameters from config file
diagnosis_classifier_device=$(get_param "diagnosis_classifier_device")
signal_processing_device=$(get_param "signal_processing_device")
data_path=$(get_param "data_path")
batch_size=$(get_param "batch_size")
output_folder=$(get_param "output_folder")
hugging_face_api_key_path=$(get_param "hugging_face_api_key_path")
# Models per input type (signal_models, image_models), passed as "key:model,model;key:model"
models=$(grep -E '^[a-z0-9_]+_models:' heartwise.config | sed 's/[[:space:]]//g' | paste -sd';' -)
ecg_signals_path=$(get_param "ecg_signals_path")
mode=$(get_param "mode")
preprocessing_folder=$(get_param "preprocessing_folder")
preprocessing_n_workers=$(get_param "preprocessing_n_workers")
single_lead=$(get_param "single_lead")

# Overwrite mode by command line argument
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                mode="$2"
                shift 2
            else
                echo "Error: --mode requires a non-empty argument."
                usage
                return 1
            fi
            ;;
        --csv_file_name)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                csv_file_name="$2"
                shift 2
            else
                echo "Error: --csv_file requires a non-empty argument."
                usage
                return 1
            fi
            ;;
        --help|-h)
            usage
            return 0
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            usage
            return 1
            ;;
    esac
done

# Function to run the pipeline with given mode
run_pipeline() {
    local run_mode=$1
    echo "Running pipeline in $run_mode mode..."
    python main.py \
        --diagnosis_classifier_device $diagnosis_classifier_device \
        --signal_processing_device $signal_processing_device \
        --data_path $data_path"/"$csv_file_name \
        --batch_size $batch_size \
        --output_folder $output_folder \
        --hugging_face_api_key_path $hugging_face_api_key_path \
        --models "$models" \
        --ecg_signals_path $ecg_signals_path \
        --mode $run_mode \
        --preprocessing_folder $preprocessing_folder \
        --preprocessing_n_workers $preprocessing_n_workers \
        ${single_lead:+--single_lead $single_lead}
}

# Main execution based on mode
case $mode in
    preprocessing)
        run_pipeline preprocessing
        ;;
    analysis)
        run_pipeline analysis
        ;;
    full_run)
        run_pipeline full_run
        ;;
    *)
        echo "Invalid mode: $mode. Use 'preprocessing', 'analysis', or 'full_run'."
        exit 1
        ;;
esac
