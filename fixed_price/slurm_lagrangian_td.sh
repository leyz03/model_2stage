#!/bin/bash
# slurm_lagrangian_td.sh — Lagrangian time decomposition runs; RUNS is n:T:scen:price joined by + (scen like 1-3).
# Example: sbatch --export=ALL,RUNS="5:12:1-3:u4+8:8:1-3:u4",VARIANTS="u4/a4",BUDGET=600 model_2stage/fixed_price/slurm_lagrangian_td.sh
# Output: results/$SLURM_JOB_ID/lagrangian_td.jsonl, slurm_logs/fp_lag-$SLURM_JOB_ID.out
# SCRIPT=run_ref_bounds.jl runs the reference bounds instead (EXTRA="--tl 600 --rf 4-2-8"); EXTRA may add
# --tag _m6 --instdir $ROOT/model_2stage/fixed_price/instances or --phi 0.
#SBATCH --job-name=fp_lag
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -euo pipefail
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=$ROOT/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep -vE "WLSAccessID|WLSSecret|LicenseID|Academic license|^Set parameter"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date), runs: $RUNS, variants: ${VARIANTS:-u4}, budget: ${BUDGET:-600}, block_tl: ${BLOCK_TL:-60}"
for run in ${RUNS//+/ }; do
  IFS=: read -r n T sc price <<< "$run"
  echo "=== $run  $(date +%T)"
  /usr/bin/time -f "    wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/fixed_price/${SCRIPT:-run_lagrangian_td.jl} \
      --n "$n" --T "$T" --scen "$sc" --price "$price" --variants "${VARIANTS:-u4}" \
      --budget "${BUDGET:-600}" --block-tl "${BLOCK_TL:-60}" --myopic "${MYOPIC:-1}" ${EXTRA:-} 2>&1 | filt
done
echo "end $(date)"
