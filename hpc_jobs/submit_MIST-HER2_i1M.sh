#!/bin/bash
# submit_MIST-HER2_i1M.sh
# Submission wrapper for the two-part MIST-HER2 1M-iteration UGATIT training.
# Submits part 1, then submits part 2 with a Slurm dependency so that
# part 2 only starts if part 1 exits cleanly (no crash, no timeout).
#
# Usage (run from the repo root on the login node):
#   bash hpc_jobs/submit_MIST-HER2_i1M.sh
#
# To check status:
#   squeue -u $USER
#
# To cancel both jobs if needed:
#   scancel <part1_jobid> <part2_jobid>
#
# After both jobs finish, submit inference:
#   sbatch hpc_jobs/infer_MIST-HER2_i1M.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART1="$SCRIPT_DIR/train_full_MIST-HER2_i1M_part1.sh"
PART2="$SCRIPT_DIR/train_full_MIST-HER2_i1M_part2.sh"

if [ ! -f "$PART1" ]; then
    echo "ERROR: $PART1 not found."
    exit 1
fi
if [ ! -f "$PART2" ]; then
    echo "ERROR: $PART2 not found."
    exit 1
fi

echo "=== Submitting MIST-HER2 1M-iteration UGATIT training (2 parts) ==="

JOB1=$(sbatch --parsable "$PART1")
echo "  Part 1 submitted: job $JOB1  (iterations 1-500,000, constant LR)"

JOB2=$(sbatch --parsable --dependency=afterok:"$JOB1" "$PART2")
echo "  Part 2 submitted: job $JOB2  (iterations 500,001-1,000,000, LR decay)"
echo "  Part 2 will only start if part 1 exits successfully."

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Part 1 log:   tail -f \$VSC_DATA/projects/ugatit/logs/train_full_MIST-HER2_i1M_p1.$JOB1.out"
echo "Part 2 log:   tail -f \$VSC_DATA/projects/ugatit/logs/train_full_MIST-HER2_i1M_p2.$JOB2.out"
echo ""
echo "After both complete, submit inference:"
echo "  sbatch hpc_jobs/infer_MIST-HER2_i1M.sh"
