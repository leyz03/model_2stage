#!/bin/bash
# slurm_fixed_price.sh — a batch of fixed-price runs, one Gurobi WLS session per job (at most 2 jobs at once).
#   sbatch --time=06:00:00 --export=ALL,RUNS="stab:3:4:5:u4+td:3:8:3:u4",OPTS="--tl 300" model_2stage/fixed_price/slurm_fixed_price.sh
#   run = mode:n:T:nscen:price; options for every run in OPTS (never use "," inside RUNS / OPTS: sbatch --export splits on it;
#   relax-and-fix variants go as --rf 2-1-lp-pre/3-1-lp-verify).  Order RUNS small to large so pruning can fire
#   (run_fixed_price.jl skips a method that failed at a dominated size; PRUNE_JOBS="jobid ..." adds earlier jobs, PRUNE=0 disables).
#   results -> model_2stage/results/$SLURM_JOB_ID/fixed_price.jsonl
#SBATCH --job-name=fp_run
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
  IFS=: read -r mode n T ns price <<< "$run"
  echo "=== $run  $(date +%T)"
  /usr/bin/time -f "    wall %e s  maxrss %M KB" $JL $ROOT/model_2stage/fixed_price/run_fixed_price.jl \
      --mode $mode --n $n --T $T --nscen $ns --price $price ${OPTS:-} 2>&1 | filt
done
echo "end $(date)"
