#!/bin/bash
#SBATCH --job-name=ugatit_train_BCI_i1M_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/ugatit/logs/train_full_BCI_i1M_p2.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/ugatit/logs/train_full_BCI_i1M_p2.%j.err

# train_full_BCI_i1M_part2.sh
# Iterations 500,001-1,000,000 of UGATIT training on BCI.
# Resumes from the 500,000-iteration checkpoint saved by part 1.
# Linear LR decay applied across these 500,000 iterations (decay_flag=True).
#
# UGATIT resume behaviour: train() finds the highest-numbered *.pt in model/,
# which after part 1 is BCI_full_i1M_params_0500000.pt. It then runs
# range(500000, 1000001) with decay starting at step 500001. No manual
# epoch_count argument needed -- resume=True handles it automatically.
#
# DO NOT submit this manually before part 1 finishes -- use submit_BCI_i1M.sh.
# If part 1 failed or was cancelled, do not submit this script.
#
# After this job completes, next step: sbatch infer_BCI_i1M.sh

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/ugatit_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/ugatit/code/ugatit"
RESULT_DIR="$VSC_DATA/projects/ugatit/outputs/results"
DATASET="BCI_full_i1M"
BCI_SQSH="$VSC_SCRATCH/BCI-AB.sqsh"
BCI_MNT="$VSC_SCRATCH/sqsh_mnt/ugatit/BCI_full_i1M"
UGATIT_DATAROOT="$VSC_SCRATCH/sqsh_mnt/ugatit"
UGATIT_DATA_BINDS=(
    -B "$BCI_SQSH:$BCI_MNT/trainA:image-src=/trainA"
    -B "$BCI_SQSH:$BCI_MNT/trainB:image-src=/trainB"
    -B "$BCI_SQSH:$BCI_MNT/testA:image-src=/valA"
    -B "$BCI_SQSH:$BCI_MNT/testB:image-src=/valB"
)

# =========================
# MODULES
# =========================

module purge
module load calcua/2026.1

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Container ==="
echo "  $CONTAINER"
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

echo ""
echo "=== Environment ==="
apptainer exec --nv "$CONTAINER" python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== SquashFS check ==="
if [ ! -f "$BCI_SQSH" ]; then
    echo "ERROR: BCI-AB.sqsh not found: $BCI_SQSH"
    exit 1
fi
echo "  BCI-AB.sqsh found"

echo ""
echo "=== Dataset check ==="
mkdir -p "$BCI_MNT"/{trainA,trainB,testA,testB}
apptainer exec \
    "${UGATIT_DATA_BINDS[@]}" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $BCI_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $BCI_MNT/trainB | wc -l) images\"; echo \"  testA:  \$(ls $BCI_MNT/testA  | wc -l) images\"; echo \"  testB:  \$(ls $BCI_MNT/testB  | wc -l) images\""

echo ""
echo "=== Checkpoint check (part 1 must have completed) ==="
CKPT_500K="$RESULT_DIR/$DATASET/model/${DATASET}_params_0500000.pt"
if [ ! -f "$CKPT_500K" ]; then
    echo "ERROR: 500k checkpoint not found: $CKPT_500K"
    echo "Has part 1 completed successfully?"
    exit 1
fi
echo "  500k checkpoint found:"
find "$RESULT_DIR/$DATASET/model" -name "*.pt" | sort

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/ugatit/logs/gpu_train_full_BCI_i1M_p2.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting BCI training part 2 (iterations 500,001-1,000,000, LR decay) ==="
echo "  dataset   : $DATASET"
echo "  dataroot  : $UGATIT_DATAROOT"
echo "  result_dir: $RESULT_DIR"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    "${UGATIT_DATA_BINDS[@]}" \
    "$CONTAINER" \
    python main.py \
        --phase       train \
        --light       True \
        --dataset     "$DATASET" \
        --dataroot    "$UGATIT_DATAROOT" \
        --iteration   1000000 \
        --save_freq   250000 \
        --print_freq  1000000 \
        --decay_flag  True \
        --resume      True \
        --img_size    512 \
        --load_size   1024 \
        --batch_size  1 \
        --result_dir  "$RESULT_DIR"

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Post-run checkpoint check ==="
find "$RESULT_DIR/$DATASET/model" -name "*.pt" | sort

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/ugatit/logs/gpu_train_full_BCI_i1M_p2.csv"

echo ""
echo "BCI full training complete (iterations 1-1,000,000). Next step: sbatch infer_BCI_i1M.sh"
