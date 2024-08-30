#!/bin/bash

# Function to get value of a parameter from config file
get_param() {
    grep "^$1:" heartwise.config | cut -d':' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Read parameters from config file
device=$(get_param "device")
data_path=$(get_param "data_path")
model_name=$(get_param "model_name")
batch_size=$(get_param "batch_size")
output_file=$(get_param "output_file")
output_folder=$(get_param "output_folder")
signal_processing_model_name=$(get_param "signal_processing_model_name")
diagnosis_classifier_model_name=$(get_param "diagnosis_classifier_model_name")

# Run the pipeline with the parameters
python main.py \
    --device $device \
    --batch_size $batch_size \
    --output_file $output_file \
    --data_path $data_path \
    --output_folder $output_folder \
    --signal_processing_model_name $signal_processing_model_name \
    --diagnosis_classifier_model_name $diagnosis_classifier_model_name