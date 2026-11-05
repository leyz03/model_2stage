#!/bin/bash
# slurm_alt_price.sh — TODO #4 runs (int_flow/run_alt_price.jl): one array task per (T, start), the start being a
# uniform price level (alternating from it) or "ub" (the upper bound only); Gurobi from the Savio site license (token
# server), so many tasks can run at once.  summarize_alt_price.py merges the tasks of a job.
#   sbatch --array=0-27 model_2stage/int_flow/slurm_alt_price.sh                  T = 10, 20 x starts 0..12 and ub
#   sbatch --array=0-13 --export=ALL,TS=50 model_2stage/int_flow/slurm_alt_price.sh
#   task i: T = TS[i / #STARTS], start = STARTS[i % #STARTS]; options for every task in OPTS
#   results -> model_2stage/int_flow/results/$SLURM_ARRAY_JOB_ID/
#SBATCH --job-name=ap
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%A_%a.out
set -u
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=/global/software/site/license/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep --line-buffered -viE "WLSACCESSID|WLSSECRET|LICENSEID"; }
read -r -a TSA <<< "${TS:-10 20}"
read -r -a SA <<< "${STARTS:-0 1 2 3 4 5 6 7 8 9 10 11 12 ub}"
i=$SLURM_ARRAY_TASK_ID
T=${TSA[$((i / ${#SA[@]}))]}
s=${SA[$((i % ${#SA[@]}))]}
if [ "$s" = ub ]; then args="--ub only"; else args="--starts $s --uniform starts --ub 0"; fi
echo "node $(hostname), job ${SLURM_ARRAY_JOB_ID}_$i ($SLURM_JOB_ID), start $(date), T $T, $args, opts: ${OPTS:-}"
/usr/bin/time -f "wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/int_flow/run_alt_price.jl \
    --T $T $args --threads $SLURM_CPUS_PER_TASK ${OPTS:-} 2>&1 | filt
echo "end $(date)"
