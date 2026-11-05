#!/bin/bash
# slurm_alt_price_test.sh — correctness tests of int_flow/alt_price.jl (test/test_alt_price.jl, Gurobi from the Savio
# site license).
#   sbatch model_2stage/int_flow/slurm_alt_price_test.sh
#SBATCH --job-name=ap_test
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -u
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=/global/software/site/license/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep --line-buffered -viE "WLSACCESSID|WLSSECRET|LICENSEID"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date)"
$JL $ROOT/model_2stage/test/test_alt_price.jl gurobi 2>&1 | filt
echo "end $(date)"
