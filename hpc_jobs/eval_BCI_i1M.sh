#!/bin/bash
#SBATCH --job-name=ugatit_eval_BCI_i1M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/ugatit/logs/eval_BCI_i1M.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/ugatit/logs/eval_BCI_i1M.%j.err

# eval_BCI_i1M.sh
# Runs evaluate.py on UGATIT BCI inference outputs using the shared evaluate_nvidia.sif.
#
# UGATIT is unpaired so --match_by sort is used: both pred and GT folders are sorted
# alphabetically and matched by position. testA images are processed in alphabetical
# order by UGATIT's dataloader (shuffle=False), so fake_B_0001.png corresponds to
# the first sorted testB image, fake_B_0002.png to the second, etc.
#
# GT images come from testB inside BCI-AB-test.sqsh mounted as BCI_full_i1M.
# Predictions come from the inference output folder (on $VSC_DATA, no sqsh needed).
#
# Submit ONLY after infer_BCI_i1M.sh has completed and image count is 977.
# Submit: sbatch eval_BCI_i1M.sh
#
# Results appended to:
#   $VSC_DATA/benchmark_results.csv

set -euo pipefail

EVAL_CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
RESULT_DIR="$VSC_DATA/projects/ugatit/outputs/results"
DATASET="BCI_full_i1M"
PRED_DIR="$RESULT_DIR/$DATASET/test"
BCI_TEST_SQSH="$VSC_SCRATCH/BCI-AB-test.sqsh"
BCI_TEST_MNT="$VSC_SCRATCH/sqsh_mnt/ugatit/BCI_full_i1M"
GT_DIR="$BCI_TEST_MNT/testB"

# =========================
# MODULES
# =========================

module purge
module load calcua/2026.1

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Evaluate container ==="
if [ ! -f "$EVAL_CONTAINER" ]; then
    echo "ERROR: evaluate_nvidia.sif not found: $EVAL_CONTAINER"
    exit 1
fi
echo "  found"

echo ""
echo "=== SquashFS check ==="
if [ ! -f "$BCI_TEST_SQSH" ]; then
    echo "ERROR: BCI-AB-test.sqsh not found: $BCI_TEST_SQSH"
    exit 1
fi
echo "  BCI-AB-test.sqsh found"

echo ""
echo "=== Prediction folder check ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction folder not found: $PRED_DIR"
    echo "Has infer_BCI_i1M.sh completed successfully?"
    exit 1
fi
echo "  fake_B images: $(find "$PRED_DIR" -name "*.png" | wc -l)"

# =========================
# EVALUATION
# =========================

mkdir -p "$BCI_TEST_MNT"

echo ""
echo "=== Starting BCI evaluation ==="
echo "  predictions : $PRED_DIR"
echo "  ground truth: $GT_DIR (inside BCI-AB-test.sqsh mounted as BCI_full_i1M)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$BCI_TEST_SQSH:$BCI_TEST_MNT:image-src=/" \
    "$EVAL_CONTAINER" \
    python "$VSC_DATA/evaluate/evaluate.py" \
        --pred           "$PRED_DIR" \
        --gt             "$GT_DIR" \
        --model_name     UGATIT \
        --dataset_name   BCI \
        --split_name     val \
        --match_by       sort \
        --output         "$VSC_DATA/benchmark_results.csv" \
        --cellpose \
        --cellpose_model cpsam \
        --cellpose_n     100

# =========================
# POST-RUN REPORT
# =========================

echo ""
echo "=== benchmark_results.csv (last 3 rows) ==="
tail -3 "$VSC_DATA/benchmark_results.csv"

echo ""
echo "BCI evaluation complete."
