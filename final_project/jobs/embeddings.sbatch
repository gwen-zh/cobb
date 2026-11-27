#!/bin/bash
#SBATCH --job-name=pg-embeddings
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embeddings-%j.out
#SBATCH --error=logs/embeddings-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
project_python=${PROJECT_GPU_PYTHON:-python}
if [[ -d "${PWD}/cache/transformers" ]]; then
    export PYTHONPATH="${PWD}/cache/transformers${PYTHONPATH:+:${PYTHONPATH}}"
fi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$project_python" -u src/embed.py --assay PHOT --pilot 8 --device cuda
"$project_python" -u src/embed.py --device cuda --batch-size 32
