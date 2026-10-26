#!/bin/bash
# Monitor WCR dual-head training for epoch 2 completion, then export ONNX
set -e

LOG="/volume/DeepECG_Docker/logs/train_dual_head_20260306_224241.log"
EXPORT_SCRIPT="/volume/DeepECG_Docker/scripts/export_dual_head_onnx.py"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring for epoch 2 completion..."

while true; do
    # Look for the epoch 2 summary line (pattern: "Epoch 2/50 |")
    if grep -q "Epoch 2/50 |" "$LOG" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Epoch 2 complete!"
        grep "Epoch 2/50 |" "$LOG"
        
        # Also grab the latest best AUC
        BEST_LINE=$(grep "New best" "$LOG" | tail -1)
        echo "Latest best: $BEST_LINE"
        break
    fi
    
    # Show progress
    LATEST=$(grep "Step " "$LOG" | tail -1)
    echo "[$(date '+%H:%M')] Waiting... $LATEST"
    sleep 300  # check every 5 min
done

echo ""
echo "=== Exporting best model to ONNX ==="
cd /volume/DeepECG_Docker
python scripts/export_dual_head_onnx.py \
    --weights checkpoints_acs_dual_head/best_model.pt \
    --output /volume/Open-ECG-Digitizer/weights/exported/acs_dual_head.onnx

echo ""
echo "=== Export complete ==="
