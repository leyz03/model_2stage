#!/bin/bash
# slurm_decomp.sh — TODO #3 runs (int_flow/run_decomp.jl), one run per array task; Gurobi from the Savio site license
# (token server), so many tasks can run at once.
#   sbatch --array=0-47 model_2stage/int_flow/slurm_decomp.sh          days 11 18 1 x T 10 20 50 100 x 4 methods
#   sbatch --array=0-5 --export=ALL,METHODS="benders",TS="10 20" model_2stage/int_flow/slurm_decomp.sh
#   task i: method = METHODS[i % #M], T = TS[(i / #M) % #TS], day = DAYS[i / (#M #TS)]; options for every run in OPTS
#   a method written m/D runs --method m --drop D (TODO #8), e.g. METHODS="full/S lp/S nox"
#   results -> model_2stage/int_flow/results/$SLURM_ARRAY_JOB_ID/
#SBATCH --job-name=dc
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%A_%a.out
set -u
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=/global/software/site/license/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep --line-buffered -viE "WLSACCESSID|WLSSECRET|LICENSEID"; }
read -r -a MA <<< "${METHODS:-full lp cg benders}"
read -r -a TSA <<< "${TS:-10 20 50 100}"
read -r -a DA <<< "${DAYS:-11 18 1}"
i=$SLURM_ARRAY_TASK_ID
nm=${#MA[@]}
method=${MA[$((i % nm))]}
drop=""
[[ $method == */* ]] && drop="--drop ${method#*/}" && method=${method%/*}
T=${TSA[$(((i / nm) % ${#TSA[@]}))]}
day=${DA[$((i / (nm * ${#TSA[@]})))]}
echo "node $(hostname), job ${SLURM_ARRAY_JOB_ID}_$i, start $(date), day $day T $T method $method $drop, opts: ${OPTS:-}"
/usr/bin/time -f "wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/int_flow/run_decomp.jl \
    --day $day --T $T --method $method $drop --threads $SLURM_CPUS_PER_TASK ${OPTS:-} 2>&1 | filt
echo "end $(date)"
