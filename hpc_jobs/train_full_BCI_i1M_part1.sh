#!/bin/bash
#SBATCH --job-name=ugatit_train_BCI_i1M_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/ugatit/logs/train_full_BCI_i1M_p1.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/ugatit/logs/train_full_BCI_i1M_p1.%j.err

# train_full_BCI_i1M_part1.sh
# Iterations 1-500,000 of UGATIT training on BCI at 512x512 (cropped from 1024).
# Constant LR throughout (decay_flag=False).
#
# This is the first half of the full 1,000,000-iteration run. The LR schedule
# is equivalent to a single 1M-iteration run:
#   Part 1: iters     1-500,000  constant LR     (decay_flag=False)
#   Part 2: iters 500,001-1,000,000  linear decay (decay_flag=True, resume=True)
#
# DO NOT submit this manually -- use submit_BCI_i1M.sh which chains both parts.
#
# Set wall time above using the validate job log:
#   (sec_per_1000iter * 500 * 1.20) / 3600 rounded up to next hour.
#
# Checkpoints saved at 250,000 and 500,000 iterations:
#   $VSC_DATA/projects/ugatit/outputs/results/BCI_full_i1M/model/
# Latest weights written every 1000 iterations:
#   $VSC_DATA/projects/ugatit/outputs/results/BCI_full_i1M_params_latest.pt

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

mkdir -p "$RESULT_DIR"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/ugatit/logs/gpu_train_full_BCI_i1M_p1.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting BCI training part 1 (iterations 1-500,000, constant LR) ==="
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
        --iteration   500000 \
        --save_freq   250000 \
        --print_freq  500000 \
        --decay_flag  False \
        --resume      False \
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
tail -3 "$VSC_DATA/projects/ugatit/logs/gpu_train_full_BCI_i1M_p1.csv"

echo ""
echo "BCI part 1 complete (iterations 1-500,000). Part 2 should start automatically if submitted via wrapper."
