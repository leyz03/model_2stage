#!/bin/bash
# slurm_decomp_test.sh — correctness tests of int_flow/decomp.jl (test/test_decomp.jl, Gurobi from the Savio site
# license).
#   sbatch model_2stage/int_flow/slurm_decomp_test.sh
#   sbatch --export=ALL,TEST=test_gap_source.jl model_2stage/int_flow/slurm_decomp_test.sh     TODO #8 variants
#SBATCH --job-name=dc_test
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
export GRB_LICENSE_FILE=/global/software/site/license/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep --line-buffered -viE "WLSACCESSID|WLSSECRET|LICENSEID"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date)"
$JL $ROOT/model_2stage/test/${TEST:-test_decomp.jl} 2>&1 | filt
echo "end $(date)"
