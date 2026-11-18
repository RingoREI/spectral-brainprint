# 验证状态与历史记录

更新日期：2026-11-18。代码与结果基准为已推送的 `5ce986c`。本文汇总已有验证证据；本轮更新文档，没有重新训练或重跑数值实验。

## 当前验证状态

| 项目 | 当前状态 | 证据与范围 |
|---|---|---|
| 四任务 LDA 对照 | 已完成并推送 | EC、EO、Imagery、SSVEP × 3 组特征 × 3 折，共 36 次训练 |
| 历史指标复现 | 通过 | 36 个案例的 accuracy、balanced accuracy、macro-F1 与历史参考在 `1e-12` 内一致；逐记录重算核对的最大差为 0 |
| 验证记录对齐 | 通过 | 同折三组特征使用相同验证记录与标签；训练/验证记录互斥；每折均覆盖 20 个注册身份 |
| 合并验证预测 | 通过 | 跨折、跨任务共 11,304 条不重复记录，覆盖当前完整有标签 Train；总体指标由预测合并后计算 |
| 训练内变换 | 已实现并测试 | 填补、标准化和 LDA 在训练折拟合；不使用无标签 Test 选择模型或阈值 |
| 结果整理与图表 | 已完成 | 总报告、12 行完整汇总指标、逐折/逐身份表、配对错误、混淆矩阵，以及 PNG/PDF 图表 |
| 自动化检查 | 最近发布前 13 项通过 | 覆盖输入对齐、闭集与泄漏约束、数值性质、任务传递、默认 EC 兼容、指标重算及跨折重复检查 |
| 本地结果归并 | 已完成 | 100 个既有文件移入 `artifacts/lda/`，移动前后大小与 SHA256 一致；练习与合成示例不重复计入正式实验 |
| GitHub 同步 | LDA 版本已推送 | `main` 已同步 `5ce986c`；上传代码、配置、文档和汇总结果，逐样本结果仍在本地 |
| 团队神经模型 | 模型定义与冻结推理适配已迁入 | 尚未在当前仓库完成真实权重、金样、训练、选模与评估的完整复现 |
| A36 非周期特征 | 尚未完成交付验收 | 团队向量适配器已有对应字段；独立特征矩阵的记录映射、列定义和质量仍需确认 |

实验设置、完整指标与解释统一见 [LDA Experiment Report](../LDA_EXPERIMENT_REPORT.md)，未舍入数据见[四任务指标 CSV](results/lda-within-task-metrics.csv)。各任务来源清单位于 `docs/results/lda-<task>-within-task/provenance.json`。

## 结论能够覆盖的范围

当前实证覆盖同一有标签会话中的注册身份识别，使用 `within_task_blocked_provisional` 文件编号分块协议。记录 ID 无交集的检查已经通过，真实 Run、连续记录来源、窗口起点与重叠关系仍待核实，不能由此声称采集来源独立。

四个任务分别训练和验证，检验的是任务条件下特征收益的一致性。现有折已经被查看，继续用于开发比较；跨任务迁移、跨会话稳定性和最终独立评估需要各自的数据与协议。三折波动和神经模型的多种子波动分别反映不同来源，不能当作独立采集重复或额外样本。

以下工作尚未完成：

- A36 或背景校正峰参数的增量实验；当前 P108 是频带功率，包含非周期背景。
- 当前仓库内的同输入 LDA/MLP 对照、原始 EEG CNN 与显式谱特征融合消融。
- 真实团队权重的端到端金样复现、独立阈值校准与开放集指标。
- GNN、真实攻击、跨会话实验，以及全新依赖环境的完整发布验收。
- 历史全部 28 个案例 × 16 种表示的重训练；本次正式复现范围为四任务三表示的 36 次训练。

## 结果目录与来源

| 位置 | 内容 |
|---|---|
| `artifacts/lda/runs/<task>/` | 每任务九个逐次结果 JSON 与 `summary.csv`，含本地逐记录预测 |
| `artifacts/lda/reports/<task>/` | 各任务生成的表格、图表与来源清单 |
| `artifacts/lda/overview/` | 四任务汇总指标与总览图 |
| `artifacts/lda/learning/`、`demo/` | 早期 EC 手工练习与合成工程示例 |
| `artifacts/lda/migration_manifest.json` | 100 个文件的迁移路径、大小与 SHA256 |
| `docs/results/` | 进入版本管理的汇总表、图表和来源清单 |
| `local/` | 早期迁移核验报告、源码快照及旧文档归档 |

`artifacts/` 和 `local/` 被 Git 忽略。原始 EEG、逐记录特征矩阵、模型权重与逐样本预测未随已发布 LDA 提交上传。原始数据与衍生材料的分享范围、贡献署名及许可证仍需团队确认。

## 环境记录

四任务 LDA 的记录环境为 Python 3.13.17、NumPy 2.5.5、SciPy 1.18.1、scikit-learn 1.9.1、threadpoolctl 3.8.0；绘图使用 Matplotlib 3.11.2。各任务来源清单保存实际版本。

[validated-current.txt](../environments/validated-current.txt) 记录的是 **2026-11-12 初始迁移验证环境**：Python 3.10.10、NumPy 2.2.6、SciPy 1.14.1、scikit-learn 1.7.2、h5py 3.16.0、threadpoolctl 3.6.0。它是直接依赖记录，并非当前 LDA 的环境锁文件。[legacy-requirements.txt](../environments/legacy-requirements.txt) 是原交接环境记录；现有结果不等于已经恢复原团队的全部环境。

## 重要历史节点

| 日期 | 已完成事项 | 保留的限制 |
|---|---|---|
| 2026-11-12 | 冻结 20 个来源 Python 文件；核对交接包 48 个文件哈希/大小；检查 Train/Test 对齐、V4 九块与 28 个案例的记录划分 | 当时没有全表重训练；主键互斥不等于真实来源组互斥 |
| 2026-11-12 | 对 3 条合成与 3 条真实记录做迁移回归，PSD、功率与九块描述符最大绝对差为 0；模拟端到端和首个 EC 案例通过 | 仅为小样本数值回归；合成准确率不作真实脑纹证据 |
| 2026-11-12 | 构建 wheel 并在独立 venv 安装、从仓库外运行模拟 CLI | 复用了已有依赖，不是全新依赖环境验证 |
| 2026-11-14 | 从 `Brainprint_ML` 接入模型定义、705 维适配及 MLP20 推理接口 | 未据此建立真实深度模型性能结论 |
| 2026-11-17 至 18 | 完成四任务 36 次 LDA 训练、结果重算和图表；统一英文总报告及本地结果目录 | 保留同会话、任务内开发协议的范围 |
| 2026-11-18 | 发布前 13 项测试与暂存区检查通过，将 LDA 更新推送为 `5ce986c` | 发布内容不包含原始数据或逐样本运行结果 |

初始迁移的详细证据仍保存在 `local/source_manifest.json`、`handoff_verification.json`、`migration_regression.json` 等本地文件中。源码对应关系见[资产清单](code_inventory.csv)。

## 复查入口

从仓库根目录执行；以下是复查命令，不表示本轮文档更新再次执行了这些检查。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/summarize_lda_results.py --input-dir artifacts/lda/runs/ec --output-dir artifacts/lda/reproductions/validation-check/ec/report
```

选择新的输出目录。训练与图表的完整复现命令见 [LDA Experiment Report](../LDA_EXPERIMENT_REPORT.md)；模型交接状态与下一步大纲见[团队协作说明](08_team_model_integration.md)。
