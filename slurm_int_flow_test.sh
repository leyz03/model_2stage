#!/bin/bash
# slurm_int_flow_test.sh — correctness tests of the integer rider split (test/test_int_flow.jl, HiGHS) and the old
# sparse / smoke tests it must not break.
#   sbatch [--export=ALL,SKIP_OLD=1,INST=<int_flow/instances json>,TW=<periods>] /global/home/users/zjl0512/sddip/model_2stage/slurm_int_flow_test.sh
#SBATCH --job-name=ts_intflow
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -u
ROOT=/global/home/users/zjl0512/sddip
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
echo "node $(hostname), job $SLURM_JOB_ID, start $(date)"
echo "### test_int_flow.jl"
$JL $ROOT/model_2stage/test/test_int_flow.jl paris=$ROOT/model_2stage/int_flow/instances/${INST:-paris_n30_H10_int.json} tw=${TW:-6}
if [ "${SKIP_OLD:-0}" != 1 ]; then        # SKIP_OLD=1: only the integer-split tests
    echo "### smoke_test.jl"
    $JL $ROOT/model_2stage/test/smoke_test.jl 2 3
    echo "### test_sparse.jl"
    $JL $ROOT/model_2stage/test/test_sparse.jl
fi
echo "end $(date)"
