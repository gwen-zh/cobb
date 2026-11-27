#!/bin/bash
#SBATCH --job-name=pg-benchmark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/benchmark-%j.out
#SBATCH --error=logs/benchmark-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
project_python=${PROJECT_CPU_PYTHON:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
"$project_python" -u src/benchmark.py --pilot --workers 1
"$project_python" -u src/benchmark.py --workers "${SLURM_CPUS_PER_TASK}"
"$project_python" -u src/summarize.py
