#!/bin/bash -l
#SBATCH --job-name=cpd_mlp_semillas
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=slurm_semillas_%j.out
#SBATCH --error=slurm_semillas_%j.err

set -euo pipefail
module purge
module load gnu12/12.4.0 python3/3.13.2
source "$HOME/cpd-env/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
echo "nodo: $(hostname)  cpus: $SLURM_CPUS_PER_TASK"
python3 bench.py --ns 10000 --ps 1,2,4,8,16,32 --qs 32,64,128,256 --reps 3 --solo caso_q --out resultados_semillas.csv
python3 analisis.py --csv resultados_semillas.csv --out figs_semillas
