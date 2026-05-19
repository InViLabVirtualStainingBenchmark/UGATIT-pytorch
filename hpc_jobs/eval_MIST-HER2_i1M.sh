#!/bin/bash
#SBATCH --job-name=ugatit_eval_MIST-HER2_i1M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21212/projects/ugatit/logs/eval_MIST-HER2_i1M.%j.out
#SBATCH -e /data/antwerpen/212/vsc21212/projects/ugatit/logs/eval_MIST-HER2_i1M.%j.err

# eval_MIST-HER2_i1M.sh
# Runs evaluate.py on UGATIT MIST-HER2 inference outputs using the shared evaluate_nvidia.sif.
#
# UGATIT is unpaired so --match_by sort is used: both pred and GT folders are sorted
# alphabetically and matched by position. testA images are processed in alphabetical
# order by UGATIT's dataloader (shuffle=False), so fake_B_0001.png corresponds to
# the first sorted testB image, fake_B_0002.png to the second, etc.
#
# GT images come from valB inside MIST-HER2.sqsh, bound as testB for UGATIT.
# Predictions come from the inference output folder (on $VSC_DATA, no sqsh needed).
#
# Submit ONLY after infer_MIST-HER2_i1M.sh has completed and image count is 1000.
# Submit: sbatch eval_MIST-HER2_i1M.sh
#
# Results appended to:
#   $VSC_DATA/benchmark_results.csv

set -euo pipefail

EVAL_CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
RESULT_DIR="$VSC_DATA/projects/ugatit/outputs/results"
DATASET="MIST-HER2_full_i1M"
PRED_DIR="$RESULT_DIR/$DATASET/test"
MIST_SQSH="$VSC_SCRATCH/MIST-HER2.sqsh"
MIST_MNT="$VSC_SCRATCH/sqsh_mnt/ugatit/MIST-HER2_full_i1M"
GT_DIR="$MIST_MNT/testB"
UGATIT_GT_BIND=(-B "$MIST_SQSH:$MIST_MNT/testB:image-src=/valB")

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
if [ ! -f "$MIST_SQSH" ]; then
    echo "ERROR: MIST-HER2.sqsh not found: $MIST_SQSH"
    exit 1
fi
echo "  MIST-HER2.sqsh found"

echo ""
echo "=== Prediction folder check ==="
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: Prediction folder not found: $PRED_DIR"
    echo "Has infer_MIST-HER2_i1M.sh completed successfully?"
    exit 1
fi
echo "  fake_B images: $(find "$PRED_DIR" -name "*.png" | wc -l)"

# =========================
# EVALUATION
# =========================

mkdir -p "$MIST_MNT/testB"

echo ""
echo "=== Starting MIST evaluation ==="
echo "  predictions : $PRED_DIR"
echo "  ground truth: $GT_DIR (valB inside MIST-HER2.sqsh)"

srun apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    "${UGATIT_GT_BIND[@]}" \
    "$EVAL_CONTAINER" \
    python "$VSC_DATA/evaluate/evaluate.py" \
        --pred           "$PRED_DIR" \
        --gt             "$GT_DIR" \
        --model_name     UGATIT \
        --dataset_name   MIST-HER2 \
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
echo "MIST evaluation complete."
