#!/bin/bash
# slurm_ef.sh — two-stage EF runs, one Gurobi WLS session per job (at most 2 jobs at once).
#   sbatch --time=06:00:00 --export=ALL,RUNS="toy-A:3:type:100+toy-A:3:od:100" /global/home/users/zjl0512/sddip/model_2stage/slurm_ef.sh
#   run = variant:levels:structure:nscen, runs joined by "+"; results -> model_2stage/results/$SLURM_JOB_ID/
#SBATCH --job-name=ts_ef
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
filt() { grep -vE "WLSAccessID|WLSSecret|LicenseID|Academic license"; }
echo "node $(hostname), job $SLURM_JOB_ID, start $(date), runs: $RUNS"
for run in ${RUNS//+/ }; do
  IFS=: read -r var lev str ns <<< "$run"
  $JL $ROOT/model_2stage/run_ef.jl --variant $var --levels $lev --structure $str --nscen $ns \
      --neval ${NEVAL:-1000} --tl ${TL:-3600} --threads 4 2>&1 | filt
done
echo "end $(date)"
