# COBB 2035

## Final project · Structure and mutational-effect prediction

We tested when protein structure improves predictions of single amino-acid substitutions beyond a sequence-only language model. The benchmark uses 6,000 measured variants from four ProteinGym assays. It compares frozen ESM-2 and SaProt embeddings and conventional structural features with ridge regression, gradient-boosted trees, and a regularized MLP. Training budgets range from 50 to 500 labels. Evaluation includes random splits and splits that hold out entire residue positions.

With 500 labels and held-out positions, adding conventional structural features to the ESM-2 baseline increased mean Spearman ρ by **0.082–0.101** across the three regressors. Adding SaProt to the same baseline changed ρ by **−0.009 to +0.009**. The best held-out score was **0.403** for gradient-boosted trees with ESM-2 and conventional structure.

| What to open | Path |
| --- | --- |
| Methods, dependencies, and how to rerun | [final_project/README.md](final_project/README.md) |
| Data preparation, embeddings, training, and plotting code | [final_project/src/](final_project/src/) |
| Processed assay and structure inputs | [final_project/data/processed/](final_project/data/processed/) |
| Scores, split definitions, and test predictions | [final_project/results/tables/](final_project/results/tables/) |
| Five figures in PNG and PDF | [final_project/results/figures/](final_project/results/figures/) |
| Written report and 48 × 36 in poster | [Report PDF](final_project/results/final_report.pdf) · [Poster PDF](final_project/results/poster_48x36in.pdf) |

Yu-Hsin Chen, Ziyi Huang, and Wenxin Zhang · Fall 2026
