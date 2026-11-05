#!/bin/bash
# slurm_tier_compare.sh — paired :pre / utility-tier fixed-price runs; RUNS is n:T:nscen:price joined by +.
# Example: sbatch --export=ALL,RUNS="3:4:2:u4+5:8:3:u4",TL=300 model_2stage/fixed_price/slurm_tier_compare.sh
# Output: results/$SLURM_JOB_ID/tier_compare.jsonl, slurm_logs/fp_tier-$SLURM_JOB_ID.out
#SBATCH --job-name=fp_tier
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -euo pipefail
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=$ROOT/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep -vE "WLSAccessID|WLSSecret|LicenseID|Academic license|^Set parameter"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date), runs: $RUNS, tl: ${TL:-300}, solver: ${SOLVER:-gurobi}"
for run in ${RUNS//+/ }; do
  IFS=: read -r n T ns price <<< "$run"
  echo "=== $run  $(date +%T)"
  /usr/bin/time -f "    wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/fixed_price/run_tier_compare.jl \
      --n "$n" --T "$T" --nscen "$ns" --price "$price" --tl "${TL:-300}" \
      --solver "${SOLVER:-gurobi}" --threads 1 2>&1 | filt
done
echo "end $(date)"
