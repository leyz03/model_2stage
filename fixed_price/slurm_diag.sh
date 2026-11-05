#!/bin/bash
# slurm_diag.sh — diag_stability.jl with HiGHS (no Gurobi session).
#   sbatch --export=ALL,ARGS="--n 3 --T 8 --price u4 --s 2" model_2stage/fixed_price/slurm_diag.sh
#SBATCH --job-name=fp_diag
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -u
ROOT=/global/home/users/zjl0512/sddip
echo "node $(hostname), job $SLURM_JOB_ID, start $(date), args: $ARGS"
IFS='+' read -ra RUNS <<< "$ARGS"
for r in "${RUNS[@]}"; do
  echo "=== $r"
  $HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env $ROOT/model_2stage/fixed_price/diag_stability.jl $r
done
echo "end $(date)"
