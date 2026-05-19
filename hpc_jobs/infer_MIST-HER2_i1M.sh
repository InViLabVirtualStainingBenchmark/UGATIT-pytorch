#!/bin/bash
#SBATCH --job-name=ugatit_infer_MIST-HER2_i1M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/ugatit/logs/infer_MIST-HER2_i1M.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/ugatit/logs/infer_MIST-HER2_i1M.%j.err

# infer_MIST-HER2_i1M.sh
# Runs inference on the full MIST-HER2 test split using the latest checkpoint
# from the MIST-HER2 1M-iteration training run.
#
# UGATIT's test() automatically loads the highest-numbered *.pt from
# results/MIST-HER2_full_i1M/model/, which after full training is the 1M checkpoint.
# Images are processed at full 1024x1024 resolution (load_size=1024, no crop).
# Output is one PNG per testA image: fake_B_NNNN.png
#
# Submit ONLY after submit_MIST-HER2_i1M.sh has completed successfully.
# Submit: sbatch infer_MIST-HER2_i1M.sh
#
# Output images land at:
#   $VSC_DATA/projects/ugatit/outputs/results/MIST-HER2_full_i1M/test/
#
# Verify after job:
#   find $VSC_DATA/projects/ugatit/outputs/results/MIST-HER2_full_i1M/test -name "*.png" | wc -l
#   Expected: 1000

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/ugatit_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/ugatit/code/ugatit"
RESULT_DIR="$VSC_DATA/projects/ugatit/outputs/results"
DATASET="MIST-HER2_full_i1M"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/ugatit/MIST-HER2_full_i1M"
UGATIT_DATAROOT="$VSC_SCRATCH/sqsh_mnt/ugatit"
UGATIT_DATA_BINDS=(
    -B "$MIST_SQSH:$MIST_MNT/trainA:image-src=/trainA"
    -B "$MIST_SQSH:$MIST_MNT/trainB:image-src=/trainB"
    -B "$MIST_SQSH:$MIST_MNT/testA:image-src=/valA"
    -B "$MIST_SQSH:$MIST_MNT/testB:image-src=/valB"
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
echo "=== Checkpoint check ==="
CKPT_DIR="$RESULT_DIR/$DATASET/model"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint folder not found: $CKPT_DIR"
    echo "Has submit_MIST-HER2_i1M.sh completed successfully?"
    exit 1
fi
echo "  Checkpoints found:"
find "$CKPT_DIR" -name "*.pt" | sort

echo ""
echo "=== Test dataset check ==="
if [ ! -f "$MIST_SQSH" ]; then
    echo "ERROR: MIST-HER2.sqsh not found: $MIST_SQSH"
    exit 1
fi
mkdir -p "$MIST_MNT"/{trainA,trainB,testA,testB}
apptainer exec \
    "${UGATIT_DATA_BINDS[@]}" \
    "$CONTAINER" \
    bash -c "echo \"  testA: \$(ls $MIST_MNT/testA | wc -l) images\"; echo \"  testB: \$(ls $MIST_MNT/testB | wc -l) images\""

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/ugatit/logs/gpu_infer_MIST-HER2_i1M.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST inference ==="
echo "  dataset   : $DATASET"
echo "  result_dir: $RESULT_DIR"
echo "  dataroot  : $UGATIT_DATAROOT (inside MIST-HER2.sqsh mounted as MIST-HER2_full_i1M)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    "${UGATIT_DATA_BINDS[@]}" \
    "$CONTAINER" \
    python main.py \
        --phase       test \
        --light       True \
        --dataset     "$DATASET" \
        --dataroot    "$UGATIT_DATAROOT" \
        --img_size    512 \
        --load_size   1024 \
        --result_dir  "$RESULT_DIR"

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID

echo ""
echo "=== Output image count ==="
find "$RESULT_DIR/$DATASET/test" -name "*.png" | wc -l

echo ""
echo "=== Output folder ==="
ls "$RESULT_DIR/$DATASET/test/" 2>/dev/null | head -5 || echo "WARNING: test/ folder not found"

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/ugatit/logs/gpu_infer_MIST-HER2_i1M.csv"

echo ""
echo "MIST inference complete. Next step: sbatch eval_MIST-HER2_i1M.sh"
