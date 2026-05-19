#!/bin/bash
#SBATCH --job-name=ugatit_validate_MIST-HER2_i200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/ugatit/logs/train_validate_MIST-HER2_i200.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/ugatit/logs/train_validate_MIST-HER2_i200.%j.err

# train_validate_MIST-HER2_i200.sh
# Runs 200 iterations of UGATIT training on MIST-HER2 as a cluster confirmation gate.
# This job must pass before submitting the full 1M-iteration training jobs.
#
# PREREQUISITES -- run ONCE on the login node before submitting this job:
#
# 1. Create MIST-HER2-test.sqsh (rename valA/valB -> testA/testB):
#      unsquashfs -d $VSC_SCRATCH/tmp_mist_test $VSC_SCRATCH/MIST-HER2.sqsh
#      mv $VSC_SCRATCH/tmp_mist_test/valA $VSC_SCRATCH/tmp_mist_test/testA
#      mv $VSC_SCRATCH/tmp_mist_test/valB $VSC_SCRATCH/tmp_mist_test/testB
#      mksquashfs $VSC_SCRATCH/tmp_mist_test $VSC_SCRATCH/MIST-HER2-test.sqsh -noappend
#      rm -rf $VSC_SCRATCH/tmp_mist_test
#
# 2. See train_validate_BCI_i200.sh for the one-time container/repo/directory setup.
#
# Submit: sbatch train_validate_MIST-HER2_i200.sh
#
# Pass criteria:
#   1. Job exits cleanly (no Python traceback in log)
#   2. Loss values in log are not NaN
#   3. Checkpoint file exists:
#        ls $VSC_DATA/projects/ugatit/outputs/validate/MIST-HER2_validate_i200/model/
#   4. GPU log has entries:
#        tail -5 $VSC_DATA/projects/ugatit/logs/gpu_train_validate_MIST-HER2_i200.csv
#   5. Record time-per-1000-iterations to estimate full-run wall time.

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/ugatit_nvidia.sif"
REPO_DIR="$VSC_DATA/projects/ugatit/code/ugatit"
RESULT_DIR="$VSC_DATA/projects/ugatit/outputs/validate"
DATASET="MIST-HER2_validate_i200"
MIST_TEST_SQSH="$VSC_SCRATCH/MIST-HER2-test.sqsh"
MIST_TEST_MNT="$VSC_SCRATCH/sqsh_mnt/ugatit/MIST-HER2_validate_i200"
UGATIT_DATAROOT="$VSC_SCRATCH/sqsh_mnt/ugatit"

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
if [ ! -f "$MIST_TEST_SQSH" ]; then
    echo "ERROR: MIST-HER2-test.sqsh not found: $MIST_TEST_SQSH"
    echo "See PREREQUISITES at the top of this script."
    exit 1
fi
echo "  MIST-HER2-test.sqsh found"

echo ""
echo "=== Dataset check ==="
mkdir -p "$MIST_TEST_MNT"
apptainer exec \
    -B "$MIST_TEST_SQSH:$MIST_TEST_MNT:image-src=/" \
    "$CONTAINER" \
    bash -c "echo \"  trainA: \$(ls $MIST_TEST_MNT/trainA | wc -l) images\"; echo \"  trainB: \$(ls $MIST_TEST_MNT/trainB | wc -l) images\"; echo \"  testA:  \$(ls $MIST_TEST_MNT/testA  | wc -l) images\"; echo \"  testB:  \$(ls $MIST_TEST_MNT/testB  | wc -l) images\""

echo ""
echo "=== Repo check ==="
if [ ! -f "$REPO_DIR/main.py" ]; then
    echo "ERROR: main.py not found in $REPO_DIR"
    exit 1
fi
echo "  main.py found"

mkdir -p "$RESULT_DIR"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/ugatit/logs/gpu_train_validate_MIST-HER2_i200.csv" & GPU_LOG_PID=$!

# =========================
# TRAINING (200 iterations -- cluster gate only)
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting MIST validate training (200 iterations) ==="
echo "  dataset   : $DATASET"
echo "  dataroot  : $UGATIT_DATAROOT (MIST-HER2-test mounted inside)"
echo "  result_dir: $RESULT_DIR"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$MIST_TEST_SQSH:$MIST_TEST_MNT:image-src=/" \
    "$CONTAINER" \
    python main.py \
        --phase       train \
        --light       True \
        --dataset     "$DATASET" \
        --dataroot    "$UGATIT_DATAROOT" \
        --iteration   200 \
        --save_freq   200 \
        --print_freq  999999 \
        --decay_flag  False \
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
find "$RESULT_DIR/$DATASET/model" -name "*.pt" 2>/dev/null | sort || echo "WARNING: no checkpoints found"

echo ""
echo "=== GPU log tail ==="
tail -3 "$VSC_DATA/projects/ugatit/logs/gpu_train_validate_MIST-HER2_i200.csv"

echo ""
echo "Validation training complete. Review the output above before submitting full runs."
echo "Record time-per-1000-iterations to estimate wall time for train_full_MIST-HER2_i1M_part1.sh."
