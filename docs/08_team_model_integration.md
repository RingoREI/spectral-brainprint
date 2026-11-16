# 团队模型接入与下一轮实验

2026-11-14。模型源码来自同级 `Brainprint_ML` 交接版。现有代码支持将冻结的 MLP20 模型用于融合 HDF5 或打包的 705 维 NPZ。

| 模型 | 仓库入口 | 输入及状态 |
|---|---|---|
| 团队 MLP20 | `models/mlp.py`、`predict-mlp20` | 705 → 256 → 128 → 20。读取三份原 `model.pt` 及匹配的 `ensemble_calibration.json`，导出 Sample,Label |
| 五分支融合模型 | `models/fusion.py` | 已归一化的分支张量和掩码、任务编号 0–3。返回 96 维嵌入、cosine、分支和专家权重 |
| ResMLP | `models/neural_baselines.py` | 已归一化的 705 维向量，作为同输入神经模型对照 |
| 原始波形 CNN | `models/neural_baselines.py` | `[N, 1, 18, 1200]`，预处理和数据划分沿用原实验约定 |

后三种提供模型定义和原权重结构兼容性。它们的训练、数据缓存与部署由 `Brainprint_ML` 历史入口提供。当前交接版包含结构、训练脚本、配置和划分，完整真实推理仍需团队提供权重和冻结校准文件。MLP20 维持各成员自己的训练折中位数与 IQR，按固定种子顺序平均 raw logits，再用最大平均 logit 与冻结阈值比较。11 月 5 日的约 3.89% 已知尾部分位与 11 月 6 日的 q05 分位属于不同配置，部署需要权重与校准来自同一批运行。多噪声工作流的 `calibration_frozen.json` 使用另一套报告结构，由其原有入口处理。

## MLP20 运行

在虚拟环境安装 `python -m pip install -e ".[deep]"`。用真实模型资产替换下面的本地路径。`--template` 可选，并按模板的 `Sample` 顺序导出。打包 NPZ 的 `x` 存原始 705 维向量，还需 `sample` 与唯一记录标识。融合 HDF5 的非周期输入为 `[N,18,2]`，每通道 `[offset,exponent]`，对应的有效性掩码为 `[N,18]`。

```powershell
python -m spectral_brainprint predict-mlp20 `
  --features local/test_vectors_raw.npz `
  --checkpoint local/seed-20261102/model.pt local/seed-20261103/model.pt local/seed-20261104/model.pt `
  --calibration local/ensemble_calibration.json `
  --template local/test_template.csv `
  --trust-legacy-checkpoint `
  --output artifacts/mlp20/predictions.csv
```

旧 Torch 档案中的 NumPy 对象依赖 pickle。`--trust-legacy-checkpoint` 仅用于团队可信来源的原档案。默认使用 PyTorch 的仅权重加载模式。输出里的拒绝数量用于检查推理行为。性能指标要用带身份标签的独立评估数据计算。

## 改进顺序

1. **冻结并复现交接版本。** 接收三份权重、列字典、归一化参数、校准文件及一条带期望 logits 的金样。用相匹配的融合数据、划分和逐样本分数重跑。用真实来源组或独立会话建下一版划分，固定训练、选模、阈值校准和留出评估的角色。缺少来源组时沿用文件编号分块，并明确同会话范围。
2. **先做功率和非周期的因子消融。** 固定对数协方差 171 维作为共同空间基线，在相同记录和划分下比较 `S`、`S+P108`、`S+A36`、`S+P108+A36`。这里 `P108` 指频带功率。历史代码把功率和协方差合称 `periodic` 279 维，直接关整个分支会同时删掉空间信息。对 MLP 用相同的输入槽位、隐藏宽度、训练预算与种子，从头重训各组。先在无伪迹输入上完成四组，再在固定伪迹块背景下做同一对照，复现全 705 维。LDA 可用相同记录和特征块作为基线。优先报告 `S+P+A` 相对 `S+P` 的非周期增益，同时报告反向功率增益、按身份和任务的结果。在同一预设误接工作点比较正确接纳，并把闭集准确率、误拒率与未知拒识率分开。特征处理与阈值分别只用对应训练和校准数据拟合，报告多种子和可用独立组的配对差值。
3. **用同一基线验证图结构的收益。** 先检查 4 秒记录内连接估计的稳定性。以 18 个电极为节点，把功率和谱参数作为节点输入，比较无边、固定电极邻接和基于逐段连接统计的边。用打乱边和将同一边特征展平给 MLP 的对照，区分连接特征本身和图传播各自的收益。是否保留图分支取决于相同协议下的增益、稳定性和计算成本。

**GitHub 是独立的发布决策。** 可以先在本地做提交，方便锁定消融所用版本。团队确认代码归属、仓库可见性与许可后，再确定目标仓库。发布内容先通过仅含代码、合成输入和说明的干净目录验证，检查训练数据、预测、权重与个人路径的跟踪情况。科研上优先共享可复现的消融结果，发布基础工具可以和实验并行准备。

## 来源对应

- `models/mlp.py` 和 `io/team_vectors.py` 对应 `Brainprint_ML/workflows/brainprint_mlp20_candidate_20261105/training.py` 的 `SimpleMLP`、`vector_input`、`vector_normalizer`，以及同目录部署代码的归一化和 logits 平均。
- `models/fusion.py` 对应 `Brainprint_ML/workflows/brainprint_team_training_20261103/brainprint/model.py` 的网络定义，也与候选及多模型实验的 `legacy/model.py` 对应。
- `models/neural_baselines.py` 对应 `Brainprint_ML/workflows/brainprint_groupcal_multimodel_20261105/architectures.py` 的模型定义。
