#!/bin/bash
# slurm_diag.sh — LP-bound diagnostic with HiGHS (no Gurobi session).
#   sbatch --export=ALL,ARGS="--n 3 --T 4+5+6+7+8+9+10+11+12 --nscen 3" model_2stage/scaling/slurm_diag.sh
#SBATCH --job-name=ts_diag
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -u
ROOT=/global/home/users/zjl0512/sddip
echo "node $(hostname), job $SLURM_JOB_ID, start $(date), args: $ARGS"
$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env $ROOT/model_2stage/scaling/diag_lp.jl $ARGS
echo "end $(date)"
