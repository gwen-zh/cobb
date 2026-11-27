# When do structural features complement protein language models?

Yu-Hsin Chen, Ziyi Huang, and Wenxin Zhang · COBB 2035 · Fall 2026

We tested whether wild-type structure improves predictions of measured single-substitution effects beyond ESM-2. We trained separate models for four ProteinGym v1.3 assays, sampling 1,500 variants from each. The panel covers PHOT fluorescence, EstA thermostability, TrpC growth-based fitness, and HMBS activity.

All models use amino-acid substitution descriptors. The additional inputs are conventional structural features (relative solvent accessibility, DSSP secondary structure, and Cα contact counts) and frozen mutant-minus-wild-type embeddings from ESM-2 35M and SaProt 35M. SaProt uses Foldseek 3Di tokens from the wild-type structure. We evaluated ridge regression, histogram gradient-boosted trees (GBT), and a regularized MLP at four training sizes (50, 100, 250, 500). For five seeds per assay, we reused the same random or position-held-out test sets across models. Preprocessing, hyperparameter selection, and early stopping used training data within each budget.

## Main result

Paired changes in mean test Spearman ρ at 500 labels with entire residue positions held out. Each comparison adds one feature block to the same substitution-descriptor + ESM-2 baseline.

| Model | Add conventional structure | Add SaProt |
| --- | ---: | ---: |
| Ridge | +0.101 | −0.009 |
| GBT | +0.082 | −0.005 |
| MLP | +0.097 | +0.009 |

The best model in this setting was GBT with ESM-2 and conventional structure (ρ = 0.403). Gains varied across assays and were largest for EstA. These results describe generalization to new positions within the same proteins for the four assays and the tested frozen representations. [Full report](REPORT.md)

## File map

| Path | Contents |
| --- | --- |
| [assays.json](assays.json) · [data/processed/](data/processed/) | Assay panel, sampled variants, wild-type reference structures, per-residue features, and structure tokens |
| [src/fetch_inputs.py](src/fetch_inputs.py) · [src/prepare.py](src/prepare.py) | Download source data and checkpoints, check residue mappings, and compute conventional structure features |
| [src/embed.py](src/embed.py) | Extract frozen ESM-2, structure-aware SaProt, and sequence-only SaProt control embeddings |
| [src/benchmark.py](src/benchmark.py) · [src/summarize.py](src/summarize.py) | Fit models with shared splits, audit test predictions, and export paired results |
| [src/plot.py](src/plot.py) · [src/publish.py](src/publish.py) | Render the scientific figures, report PDF, and poster |
| [results/tables/metrics.csv](results/tables/metrics.csv) · [results/tables/mean_paired_effects.csv](results/tables/mean_paired_effects.csv) | Per-fit results for 3,360 main fits and 160 sequence-only control fits, plus effect-size summaries |
| [results/tables/splits.json](results/tables/splits.json) · [results/tables/predictions.csv.gz](results/tables/predictions.csv.gz) · [results/tables/audit.json](results/tables/audit.json) | 40 split manifests, 1,331,616 test predictions, and leakage/score checks |
| [results/figures/](results/figures/) · [Report PDF](results/final_report.pdf) · [Poster PDF](results/poster_48x36in.pdf) | Five individual figures and the formatted project materials |
| [WALKTHROUGH_ZH.md](WALKTHROUGH_ZH.md) | Chinese walkthrough and English presentation notes |

## Rebuild the figures and PDFs

From the repository root, use the checked-in result tables and reference coordinates.

```bash
cd final_project
python -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.26' 'pandas>=2.1' 'matplotlib>=3.8' 'reportlab>=4'
python src/plot.py
python src/publish.py
```

These commands use the stored benchmark results and run on CPU. Edit [REPORT.md](REPORT.md) to update the report text. The poster layout is in [src/publish.py](src/publish.py).

## Re-run the full benchmark

Use Python 3.12 and put `mkdssp` on PATH. Run the following from `final_project/` in an activated environment. The download step fetches the source archives and model checkpoints into `cache/` and extracts the four selected assays into `data/raw/`.

```bash
python -m pip install -r requirements.txt
python src/fetch_inputs.py
python src/prepare.py
python src/embed.py --device auto --batch-size 32
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/benchmark.py --workers 4
python src/summarize.py
python src/plot.py
python src/publish.py
```

The download script provides a Linux AVX2 Foldseek executable. With a separately installed Foldseek, use `python src/fetch_inputs.py --part data E P Q` and pass its path to `python src/prepare.py --foldseek /path/to/foldseek`. Embedding inference uses CUDA when available. For Slurm, [jobs/](jobs/) contains GPU and CPU templates. Submit from `final_project/`, create `logs/` before submission, and pass the appropriate `--partition` for your cluster. The jobs use `python` on PATH or the `PROJECT_GPU_PYTHON` and `PROJECT_CPU_PYTHON` overrides.

The archived run used Python 3.12.14, PyTorch 2.6.0, Transformers 4.53.3, scikit-learn 1.9.0, Biopython 1.88, NumPy 2.5.2, SciPy 1.18.0, pandas 2.3.3, Matplotlib 3.11.1, DSSP 4.2.2, and Foldseek commit `4f5c00ccb5590f08a261b9144fd9a5d28a71fa0e`. The TrpC analysis uses the ProteinGym assay sequence and its matched reference structure. The sequence matches 251 of 252 positions in UniProt Q56319.

## Sources

Processed measurements and structures derive from [ProteinGym v1.3](https://zenodo.org/records/15293562). See [source checksums](data/provenance.json), the upstream [MIT license notice](data/PROTEINGYM_LICENSE.txt), and the assay citations in [REPORT.md](REPORT.md). Encoder checkpoints come from [ESM-2](https://github.com/facebookresearch/esm) and [SaProt](https://github.com/westlake-repl/SaProt). Codex assisted with implementation, verification, and drafting. Figures are rendered from benchmark tables and reference coordinates.
