#!/bin/bash
# Fine-tune WCR on ACS Acute Obstruction prediction
# Usage: bash scripts/run_finetune_acs.sh <manifest_dir> <save_dir> [wandb_run_name]

set -e

MANIFEST_DIR="${1:?Usage: $0 <manifest_dir> <save_dir> [wandb_run_name]}"
SAVE_DIR="${2:?Usage: $0 <manifest_dir> <save_dir> [wandb_run_name]}"
WANDB_RUN_NAME="${3:-acs_finetune}"

cd /volume/DeepECG_Docker

echo "=== ACS Fine-tuning ==="
echo "Manifest dir: $MANIFEST_DIR"
echo "Save dir: $SAVE_DIR"
echo "WandB run name: $WANDB_RUN_NAME"

WANDB_PROJECT=ecg-digitization-ft \
WANDB_RUN_NAME=$WANDB_RUN_NAME \
python3 -m fairseq_signals.train \
    --config-dir fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer \
    --config-name diagnosis_acs \
    common.fp16=true \
    common.wandb_project=ecg-digitization-ft \
    task.data=$MANIFEST_DIR \
    +task.npy_dataset=true \
    task.df_dataset=false \
    model.model_path=weights/wcr_77_classes/base_ssl.pt \
    model.num_labels=1 \
    criterion._name=binary_cross_entropy_with_logits \
    criterion.report_auc=true \
    'criterion.weights_file=null' \
    checkpoint.save_dir=$SAVE_DIR \
    checkpoint.patience=20 \
    optimization.max_epoch=100 \
    'optimization.lr=[0.00005]' \
    dataset.batch_size=64 \
    "$@"
