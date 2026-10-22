#!/bin/bash -l
#SBATCH --job-name=cpd_numa
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_numa_%j.out
#SBATCH --error=slurm_numa_%j.err

set -euo pipefail
module purge
module load gnu12/12.4.0 python3/3.13.2
source "$HOME/cpd-env/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
echo "nodo: $(hostname)  cpus: $SLURM_CPUS_PER_TASK"
(numactl --hardware 2>/dev/null || lscpu | grep -i numa) | tee numa_$(hostname).txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "governor: no visible"
python3 numa_prueba.py --n 40000 --reps 3
