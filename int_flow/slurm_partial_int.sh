#!/bin/bash
# slurm_partial_int.sh — TODO #2 runs (int_flow/run_partial_int.jl), one run per array task; Gurobi from the Savio
# site license (token server), so many tasks can run at once.
#   sbatch --array=0-47 model_2stage/int_flow/slurm_partial_int.sh          days 11 18 1 x T 10 20 50 100 x 4 forms
#   sbatch --array=0-3 --qos=savio_debug --time=00:30:00 --export=ALL,TS=4,DAYS=18 model_2stage/int_flow/slurm_partial_int.sh
#   task i: form = FORMS[i % 4], T = TS[(i / 4) % #TS], day = DAYS[i / (4 #TS)]; options for every run in OPTS
#   results -> model_2stage/int_flow/results/$SLURM_ARRAY_JOB_ID/
#SBATCH --job-name=pi
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
filt() { grep -viE "WLSACCESSID|WLSSECRET|LICENSEID"; }
FORMS=(full xbin x lp)
read -r -a TSA <<< "${TS:-10 20 50 100}"
read -r -a DA <<< "${DAYS:-11 18 1}"
i=$SLURM_ARRAY_TASK_ID
form=${FORMS[$((i % 4))]}
T=${TSA[$(((i / 4) % ${#TSA[@]}))]}
day=${DA[$((i / (4 * ${#TSA[@]})))]}
echo "node $(hostname), job ${SLURM_ARRAY_JOB_ID}_$i, start $(date), day $day T $T form $form, opts: ${OPTS:-}"
/usr/bin/time -f "wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/int_flow/run_partial_int.jl \
    --day $day --T $T --form $form --threads $SLURM_CPUS_PER_TASK ${OPTS:-} 2>&1 | filt
echo "end $(date)"
