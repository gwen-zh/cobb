# 这份项目怎么讲

我们想知道一件事。训练变体有限时，显式知道蛋白某个位置的结构，能不能让模型更准确地预测该位置突变后的实验读数？老师的两条反馈对应两个改动。我们把 SaProt 加入特征比较，把梯度提升树和正则化 MLP 加入原来的 Ridge 对照。

## 数据和方法

1. 每个 assay 对应一个蛋白、一个参考结构和一组实测 DMS scores。用 PHOT 荧光、EstA 热稳定性、TrpC 生长适应度和 HMBS 活性四类读数，分别建模。每个 assay 保留 1,500 个单点替换。
2. **D** 说明什么氨基酸换成什么。**E** 是 ESM-2 从序列看到的变化。**S** 是 WT 位点的溶剂暴露度、二级结构、邻居数。**P** 是 SaProt 从氨基酸和 WT 结构 token 一起看到的变化。E 和 P 都取突变位置的最后一层 embedding，用 mutant 减 WT。
3. 要量化 S 是否补充 E，直接用同一组训练和测试变体，比较 `D+E+S` 和 `D+E`。要量化 P 是否补充 E，则比较 `D+E+P` 和 `D+E`。用的是 Spearman ρ，即预测排名和实验排名的一致程度。同一折的两个 ρ 相减得到 Δρ。
4. 随机划分允许同一位点的不同替换落到训练和测试。例如 `A42V` 在训练而 `A42K` 在测试。按位置留出时，42 位的变体只进入一侧。每个预算单独选超参数，验证和 PCA 都在该预算的训练集内。

## 用五张图讲结果

- **图 1 整体表现。** 500 个标签时，位置留出下 `GBT + D+E+S` 最好，ρ 为 0.403。相同 GBT 用 `D+E` 时是 0.321。传统结构也改善了 Ridge 和 MLP，所以增益在本实验里跨三种模型都能看到。
- **图 2 学习曲线。** 纵轴是相对 `D+E` 的配对增益。绿色代表加传统结构，橙色代表加 SaProt。位置留出的绿色线在各预算下平均为正。橙色接近零，在随机划分的 500 标签下才更有优势。
- **图 3 埋藏和暴露。** 只比较两个亚组都够大的测试折。传统结构的点估计未集中于埋藏位点，误差条体现了不同划分的波动。TrpC 留出的暴露位点太少，放进两组会破坏对照。
- **图 4 随机与位置留出的差距。** 同样 500 标签，随机划分的均分更高。加 SaProt 后差距变大，说明在熟悉位点的表现不能代替未见位点的表现。
- **图 5 EstA 实例。** S158Q 的实测排名是第 9 百分位，加结构后预测从 94 移到 20，靠近实测。G135M 的实测是第 100 百分位，加结构后从 68 移到 2，反而更远。两个位点的 RSA 都是 0，说明同属埋藏类别时仍可能出现不同预测误差。

## 90 秒英文讲稿

> We asked when structural information improves prediction of single-substitution effects beyond a sequence language model. We benchmarked 6,000 measured variants across four ProteinGym assays. For each assay, we compared frozen ESM-2 with conventional structural features and frozen SaProt embeddings, using ridge regression, gradient-boosted trees, and a small regularized MLP.
>
> Our most important comparison holds out entire residue positions. With 500 training labels, adding solvent accessibility, secondary structure, and contact counts increased mean Spearman correlation by 0.082 to 0.101 across the three models. Gradient boosting with ESM-2 and conventional structure reached 0.403. Adding SaProt to the ESM-2 baseline made little difference on held-out positions, even though it improved the random-split means at this budget.
>
> The gains depended strongly on the assay. EstA benefited the most, and TrpC did slightly worse. Two buried EstA examples also changed in opposite directions. Overall, explicit geometry was the more consistent addition for unseen sites on this panel. The SaProt conclusion is specific to the frozen 35-million-parameter model, our embedding reduction, and the four assays we tested.

## 可能被追问

**Why hold out positions?** 相同位点的野生型、接触数和二级结构是共享的。随机划分会让模型有机会从某一位点的已测替换推断同一位点的其他替换。位置留出检验更严格的位点泛化。

**Why did SaProt add little?** 两套预训练表征可能共享信号。SaProt 只使用固定 WT 的 3Di，且 480 维向量压到 24 维。现有实验能定位增益出现在哪种设置，具体原因还需要控变量实验。Ridge 的同尺寸纯序列 SaProt 对照在位置留出下比结构感知版本高 0.039。

**Does your experiment generalize to new proteins?** 这次每个 assay 单独训练，直接证据是同一蛋白中对未训练过的位点的泛化。新蛋白需要按蛋白或蛋白家族留出另做对照。

**Are the differences significant?** 图里的误差范围是五个种子得到的均值的标准差，反映划分带来的变化。测试折会重叠，四个 assay 的读数也不同，所以报告使用配对效应量和逐 assay 的趋势。
