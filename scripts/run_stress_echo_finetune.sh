#!/bin/bash
# Fine-tune WCRv2 on stress-echo latent obstruction cohorts.
# Runs both cohorts sequentially with wandb logging.
#
# Usage:
#   bash scripts/run_stress_echo_finetune.sh                 # both cohorts
#   bash scripts/run_stress_echo_finetune.sh cohort_latent_all
#   bash scripts/run_stress_echo_finetune.sh cohort_latent_vs_hcm

set -e
cd /volume/DeepECG_Docker

# fairseq-signals isn't installed as a package on this host — add its source to PYTHONPATH.
export PYTHONPATH="/volume/DeepECG_Docker/fairseq-signals:${PYTHONPATH}"

# If no wandb API key is set, fall back to offline mode (can be synced later with `wandb sync`).
if [ -z "$WANDB_API_KEY" ] && ! grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
    export WANDB_MODE=${WANDB_MODE:-offline}
    echo "[wandb] running in offline mode (no API key detected). Sync later with: wandb sync <run_dir>"
fi

COHORTS=${1:-both}
EPOCHS=${EPOCHS:-50}
LR=${LR:-0.00005}
BATCH=${BATCH:-32}
WANDB_PROJECT=${WANDB_PROJECT:-stress-echo-latent-obstruction}

run_one () {
  local NAME=$1
  local MANIFEST_DIR="/volume/DeepECG_Docker/finetune_data/stress_echo/${NAME}/manifest"
  local SAVE_DIR="/volume/DeepECG_Docker/checkpoints_stress_echo/${NAME}"
  local MODEL_PATH="/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt"
  mkdir -p "$SAVE_DIR"
  echo "=== [$NAME] manifest=$MANIFEST_DIR save=$SAVE_DIR ==="
  WANDB_PROJECT=$WANDB_PROJECT \
  WANDB_RUN_NAME=$NAME \
  python3 -m fairseq_cli.hydra_train \
      --config-dir fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer \
      --config-name diagnosis_acs \
      common.fp16=false \
      common.wandb_project=$WANDB_PROJECT \
      task.data=$MANIFEST_DIR \
      +task.npy_dataset=true \
      task.df_dataset=false \
      model.model_path=$MODEL_PATH \
      model.num_labels=1 \
      criterion._name=binary_cross_entropy_with_logits \
      criterion.report_auc=true \
      'criterion.weights_file=null' \
      checkpoint.save_dir=$SAVE_DIR \
      checkpoint.patience=10 \
      optimization.max_epoch=$EPOCHS \
      "optimization.lr=[$LR]" \
      dataset.batch_size=$BATCH \
      2>&1 | tee -a "$SAVE_DIR/train.log"
}

if [ "$COHORTS" = "both" ]; then
  run_one cohort_latent_all
  run_one cohort_latent_vs_hcm
else
  run_one "$COHORTS"
fi
