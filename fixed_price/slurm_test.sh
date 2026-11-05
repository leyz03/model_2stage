#!/bin/bash
# slurm_test.sh — correctness tests of src/fixed_price.jl (test/test_fixed_price.jl).
#   sbatch model_2stage/fixed_price/slurm_test.sh              (HiGHS, no Gurobi session)
#   sbatch --export=ALL,SOLVER=gurobi model_2stage/fixed_price/slurm_test.sh
#SBATCH --job-name=fp_test
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
filt() { grep -vE "WLSAccessID|WLSSecret|LicenseID|Academic license|^Set parameter"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date), solver ${SOLVER:-highs}"
$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env \
    $ROOT/model_2stage/test/test_fixed_price.jl ${SOLVER:-} 2>&1 | filt
echo "end $(date)"
