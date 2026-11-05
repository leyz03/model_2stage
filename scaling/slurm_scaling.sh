#!/bin/bash
# slurm_scaling.sh — a batch of scaling runs, one Gurobi WLS session per job (at most 2 jobs at once).
#   sbatch --time=08:00:00 --export=ALL,RUNS="oracle:3:4:3+bnb:5:8:50:pre" model_2stage/scaling/slurm_scaling.sh
#   run = mode:n:T:nscen[:inner], several inners joined by "/" (never ","); extra options for every run in OPTS (e.g. OPTS="--tl 1800 --mip_tl 300")
#   results -> model_2stage/results/$SLURM_JOB_ID/scaling.jsonl
#   pruning: a run is skipped when a smaller run (n'<=n, T'<=T, K_s'<=K_s, same mode/inner) already
#   failed in this job or in the jobs listed in PRUNE_JOBS (space separated); see scaling/prune.py.
#   Set PRUNE=0 to disable.
#SBATCH --job-name=ts_scale
#SBATCH --account=fc_evnetwork
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/global/home/users/zjl0512/sddip/model_2stage/slurm_logs/%x-%j.out
set -u
ROOT=/global/home/users/zjl0512/sddip
export GRB_LICENSE_FILE=$ROOT/gurobi.lic
JL="$HOME/julia-1.12.6/bin/julia --threads=1 --project=$ROOT/model_2stage/env"
filt() { grep -vE "WLSAccessID|WLSSecret|LicenseID|Academic license|^Set parameter"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date), runs: $RUNS, opts: ${OPTS:-}"
for run in ${RUNS//+/ }; do
  IFS=: read -r mode n T ns inner <<< "$run"
  extra=""; [ -n "${inner:-}" ] && extra="--inner $inner"
  echo "=== $run  $(date +%T)"
  if [ "${PRUNE:-1}" = 1 ] && msg=$(python3 $ROOT/model_2stage/scaling/prune.py $mode $n $T $ns "${inner:-}" \
         $SLURM_JOB_ID ${PRUNE_JOBS:-}); then
    echo "    $msg"; continue
  fi
  /usr/bin/time -f "    wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/scaling/run_scaling.jl \
      --mode $mode --n $n --T $T --nscen $ns $extra ${OPTS:-} 2>&1 | filt
done
echo "end $(date)"
