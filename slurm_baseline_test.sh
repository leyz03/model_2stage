#!/bin/bash
# slurm_baseline_test.sh — correctness tests of the baseline solver (one Gurobi WLS session).
#   sbatch /global/home/users/zjl0512/sddip/model_2stage/slurm_baseline_test.sh
#SBATCH --job-name=ts_btest
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -u
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=$ROOT/gurobi.lic
filt() { grep -vE "WLSAccessID|WLSSecret|LicenseID|Academic license"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date)"
$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env \
    $ROOT/model_2stage/test/test_baseline.jl 2>&1 | filt
echo "end $(date)"
