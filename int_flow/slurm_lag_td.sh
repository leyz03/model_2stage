#!/bin/bash
# slurm_lag_td.sh — TODO #10 (int_flow/run_lag_td.jl): Lagrangian time decomposition, Gurobi from the Savio site
# license (token server), so the blocks run at once.
#   PHASE=lp (one task): joint LP of the linked blocks, writes results/<job>/lambda_dDD_TTTT_BB.json
#     sbatch --export=ALL,PHASE=lp,DAY=11,T=100,B=5,OPTS="--instance <abs path>" model_2stage/int_flow/slurm_lag_td.sh
#   PHASE=block (array): task i solves block k = i % B + 1 in form FORMS[i / B] at the λ of LAM
#     sbatch --array=0-19 --dependency=afterok:<lp job> --export=ALL,PHASE=block,DAY=11,T=100,B=5,FORMS="full xbin bin lp",\
#            LAM=<abs path of the lambda json>,OPTS="--instance <abs path> --tl 1800" model_2stage/int_flow/slurm_lag_td.sh
#SBATCH --job-name=lag
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%A_%a.out
set -u
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=/global/software/site/license/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep --line-buffered -viE "WLSACCESSID|WLSSECRET|LICENSEID"; }
PHASE=${PHASE:-lp}
ARGS_RUN="--phase $PHASE --day ${DAY:-11} --T ${T:-100} --blocks ${B:-5} --threads $SLURM_CPUS_PER_TASK"
if [[ $PHASE == block ]]; then
    read -r -a FA <<< "${FORMS:-full xbin bin lp}"
    i=${SLURM_ARRAY_TASK_ID:-0}
    k=$((i % ${B:-5} + 1))
    form=${FA[$((i / ${B:-5}))]}
    ARGS_RUN="$ARGS_RUN --k $k --form $form --lam $LAM"
fi
echo "node $(hostname), job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}, start $(date), $ARGS_RUN ${OPTS:-}"
/usr/bin/time -f "wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/int_flow/run_lag_td.jl $ARGS_RUN ${OPTS:-} 2>&1 | filt
echo "end $(date)"
