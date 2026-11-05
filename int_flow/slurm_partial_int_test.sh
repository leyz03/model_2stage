#!/bin/bash
# slurm_partial_int_test.sh — correctness tests of int_flow/partial_int.jl (test/test_partial_int.jl, HiGHS).
#   sbatch model_2stage/int_flow/slurm_partial_int_test.sh
#SBATCH --job-name=pi_test
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -u
ROOT=/global/home/users/zjl0512/sddip
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
echo "node $(hostname), job $SLURM_JOB_ID, start $(date)"
$JL $ROOT/model_2stage/test/test_partial_int.jl
echo "end $(date)"
