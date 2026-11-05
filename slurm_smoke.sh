#!/bin/bash
# slurm_smoke.sh — correctness tests of model_2stage (HiGHS only, no Gurobi session).
#   sbatch /global/home/users/zjl0512/sddip/model_2stage/slurm_smoke.sh      (args: NEQ scenarios for T1/T2, NEF for T3)
#SBATCH --job-name=ts_smoke
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
echo "node $(hostname), job $SLURM_JOB_ID, start $(date)"
$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env \
    $ROOT/model_2stage/test/smoke_test.jl ${NEQ:-2} ${NEF:-3}
echo "end $(date)"
