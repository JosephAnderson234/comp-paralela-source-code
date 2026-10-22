#!/bin/bash -l
#SBATCH --job-name=cpd_mlp_beta2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_beta2_%j.out
#SBATCH --error=slurm_beta2_%j.err

set -euo pipefail
module purge
module load gnu12/12.4.0 python3/3.13.2
source "$HOME/cpd-env/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
echo "nodo: $(hostname)  cpus: $SLURM_CPUS_PER_TASK"
lscpu > lscpu_$(hostname).txt
python3 bench.py --ns 5000,10000,20000,40000 --ps 1,2,4,8,16,32 --q 32 --reps 3 --solo caso_a --version beta2 --out resultados_beta2.csv
python3 analisis.py --csv resultados_beta2.csv --out figs_beta2
