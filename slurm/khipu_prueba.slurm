#!/bin/bash -l
#SBATCH --job-name=cpd_mlp_test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm_test_%j.out

set -euo pipefail
module purge
module load gnu12/12.4.0 python3/3.13.2
source "$HOME/cpd-env/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
python3 beta1_secuencial.py --n 5000 --q 8 --E 50 --B 1024
python3 beta3_pram.py --n 5000 --q 8 --p 4 --E 50
python3 bench.py --ns 2000,5000 --ps 1,2,4,8 --q 8 --reps 1 --E 50 --out prueba.csv
python3 analisis.py --csv prueba.csv --out figs_prueba
