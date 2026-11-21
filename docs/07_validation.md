# 验证状态与历史记录

更新日期：2026-11-19。既有 LDA 与协作文档已推送至 `b72bcb6`；本轮新增 A36 恢复和特征级 MLP 消融，四任务均完成四条件、三折、三种子，共 144 次训练，并完成收敛／错误诊断与48次同训练角色LDA对照。以下明确区分当前执行结果与历史验证，新的代码和运行产物尚未随本轮更新推送。

## 当前验证状态

| 项目 | 当前状态 | 证据与范围 |
|---|---|---|
| 四任务 LDA 对照 | 已完成并推送 | EC、EO、Imagery、SSVEP × 3 组特征 × 3 折，共 36 次训练 |
| 历史指标复现 | 通过 | 36 个案例的 accuracy、balanced accuracy、macro-F1 与历史参考在 `1e-12` 内一致；逐记录重算核对的最大差为 0 |
| 验证记录对齐 | 通过 | 同折三组特征使用相同验证记录与标签；训练/验证记录互斥；每折均覆盖 20 个注册身份 |
| 合并验证预测 | 通过 | 跨折、跨任务共 11,304 条不重复记录，覆盖当前完整有标签 Train；总体指标由预测合并后计算 |
| 训练内变换 | 已实现并测试 | 填补、标准化和 LDA 在训练折拟合；不使用无标签 Test 选择模型或阈值 |
| 结果整理与图表 | 已完成 | 总报告、12 行完整汇总指标、逐折/逐身份表、配对错误、混淆矩阵，以及 PNG/PDF 图表 |
| 自动化检查 | 当前完整 37 项通过 | 包含既有 13 项，以及消融训练内变换、分组与外层隔离、结果对齐和效应量汇总等新增检查 |
| 本地结果归并 | 已完成 | 100 个既有文件移入 `artifacts/lda/`，移动前后大小与 SHA256 一致；练习与合成示例不重复计入正式实验 |
| GitHub 同步 | 既有版本已推送 | `b72bcb6` 包含 LDA 与更新后的协作文档；本轮消融尚未推送，逐样本预测与权重继续本地保存 |
| 新 315 维 MLP 消融 | 四任务 144 次训练与汇总完成 | 相同输入槽位和参数量、内层 NLL 选模、外层仅最终评估；任务内效应与四任务总览已生成 |
| 收敛与错误诊断 | 已完成 | 全部144条训练轨迹；按身份／记录／混淆对保留异质性，不据外层结果改预算 |
| 同训练角色LDA对照 | 48次拟合、完整配对与独立核对通过 | 12份预处理精确重现，系数重建预测最大概率差0，拟合警告0；不制造LDA种子重复 |
| 旧 705 维团队模型 | 定义与推理适配已迁入，原权重仍待恢复 | 不把新 315 维消融等同于历史部署模型复现；CNN 融合、ResMLP 与 GNN 尚无本轮真实对照 |
| A36 非周期特征 | 四任务 11,304 条已恢复并初步验收 | 203,472 个通道拟合无失败；保存拟合质量和负 exponent 等诊断，无分数驱动的质量删样本 |

实验设置、完整指标与解释统一见 [LDA Experiment Report](../LDA_EXPERIMENT_REPORT.md)，未舍入数据见[四任务指标 CSV](results/lda-within-task-metrics.csv)。各任务来源清单位于 `docs/results/lda-<task>-within-task/provenance.json`。

## 本轮四任务消融执行记录

- 从只读 `data_basic` 波形恢复全部 11,304 条 A36，203,472 个通道拟合无失败。配置为300 Hz、4秒、18通道、median Welch、1秒窗口、2–40 Hz拟合、FOOOF fixed模式；列顺序为offset18后exponent18。未依据身份预测分数筛除样本，质量表保留负exponent等诊断。
- 对一条 EC 记录的36个参数，新旧提取代码在当前环境的最大差为0。另独立重拟合原始H5的3条EC记录，以及EO、Imagery、SSVEP各1条记录，共108个通道谱，最大差也为0。这不等于已证明与历史完整A36矩阵相等。
- 内层角色从历史验证归属恢复三个 provisional filename block；每个外层折用剩余两个块分别作训练和开发，并保留编号边界排除。每任务三角色均覆盖20个身份，组名不跨角色；这不证明真实Run或连续窗口来源独立。
- 预处理参数已从 inner-train 独立重算核对；模型选择只访问 inner-dev。标准化后屏蔽P/A，四条件同一315槽位、117,140参数、同种子初始权重和训练预算。四任务各36次训练，合计144次，均完成汇总。

| 任务 | A_given_P均值（pp） | 种子SD（pp） | 训练达到80轮 | 最佳轮次为80 |
|---|---:|---:|---:|---:|
| EC | +0.949737 | 0.359201 | 24/36 | 22/36 |
| EO | +0.386876 | 0.148781 | 23/36 | 21/36 |
| Imagery | +0.212838 | 0.013361 | 0/36 | 0/36 |
| SSVEP | +0.591347 | 0.161370 | 10/36 | 2/36 |

三个合并种子的主要增益在各任务均为正，但EC/EO/Imagery有2/1/1个折×种子组合为负；SSVEP九个组合均为正。种子SD不是CI，不据此报告显著性。全体57/144训练达到预算上限，45/144最佳轮次位于第80轮；当前不是充分调参后的最优性能。逐次曲线和全部负结果保留。

任务内汇总位于 `artifacts/ablation/<task>/full/report/`。四任务总览由 `scripts/ablation/overview.py` 生成于 `artifacts/ablation/overview/`，包含机器可读指标、主要与次要效应、PNG/PDF及来源清单。所有源预测仍保存在本地。

## 收敛、错误与匹配角色对照：完成记录

本轮又完成48次确定性LDA拟合，均严格使用MLP的inner-train，四条件实际启用维数171/279/207/315；开发集只作保留诊断，不搜索LDA参数。均匀20类先验、lsqr与auto收缩在拟合前冻结；使用同一训练内填补、标准化和float32输入量化。完整方法在09的第12节，实际执行快照随run_manifest保存。

| 任务 | LDA：S+P+A BA | MLP：S+P+A BA 均值 | LDA 加 A 增益 | MLP 加 A 增益 |
|---|---:|---:|---:|---:|
| EC | 80.91% | 83.61% | +1.252 pp | +0.950 pp |
| EO | 76.04% | 79.04% | +0.271 pp | +0.387 pp |
| Imagery | 95.55% | 97.04% | +0.229 pp | +0.213 pp |
| SSVEP | 87.87% | 91.59% | +0.666 pp | +0.591 pp |

全部16个MLP条件的种子均值BA高于匹配LDA；A36在两类模型中的条件增益均为正，但不是MLP普遍更大。144条开发轨迹显示EC/EO仍受80轮边界限制；fold 1的平均最佳内层NLL在四任务均较高。身份平均改善／退步／不变依次为EC 11/6/3、EO 9/6/5、Imagery 10/5/5、SSVEP 14/2/4。SSVEP身份19三种子均下降（均值−5.202 pp），全部负效应保留。

37项测试通过；LDA48案例、12份训练内预处理、保存系数与概率均完成独立核对，概率最大差0、拟合警告0。固定预算开发阶段可收尾；未证明全模型充分收敛、真实来源独立或跨会话泛化。英文补充报告在本地 `artifacts/ablation/FOLLOWUP_REPORT.md`。

## 结论能够覆盖的范围

当前实证覆盖同一有标签会话中的注册身份识别，使用 `within_task_blocked_provisional` 文件编号分块协议。记录 ID 无交集的检查已经通过，真实 Run、连续记录来源、窗口起点与重叠关系仍待核实，不能由此声称采集来源独立。

四个任务分别训练和验证，检验的是任务条件下特征收益的一致性。现有折已经被查看，继续用于开发比较；跨任务迁移、跨会话稳定性和最终独立评估需要各自的数据与协议。三折波动和神经模型的多种子波动分别反映不同来源，不能当作独立采集重复或额外样本。

以下工作尚未完成：

- 背景校正峰参数尚未测试；当前P108是含非周期背景的频带功率，A36与P108也使用不同冻结PSD估计设置。
- 原始 EEG CNN 与显式谱特征融合消融。匹配内层角色LDA对照已完成；历史较大训练池的LDA仍是独立协议结果。
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
| `scripts/ablation/` | 准备、训练、汇总、学习曲线和使用说明 |
| `artifacts/ablation/ec/prepared/`、`shared/prepared_other_tasks/` | 四任务A36、固定特征、质量与内外层角色清单 |
| `artifacts/ablation/<task>/full/runs/`、`full/report/` | 各任务逐次预测、权重、曲线、效应表和PNG/PDF |
| `artifacts/ablation/overview/` | 四任务指标、主要/次要效应和总览图；由overview.py复现 |
| `artifacts/ablation/diagnostics/{convergence,errors}/` | 全轨迹与逐身份／混淆对诊断表和PNG/PDF |
| `artifacts/ablation/matched_lda/{runs,report}/` | 同训练角色的48次LDA、配对比较与PNG/PDF |
| `artifacts/ablation/FOLLOWUP_REPORT.md` | 英文阶段收尾说明与结论边界 |
| `docs/results/` | 既有 LDA 版本管理汇总表、图表和来源清单 |
| `local/` | 早期迁移核验报告、源码快照及旧文档归档 |

`artifacts/` 和 `local/` 被 Git 忽略。原始 EEG、逐记录特征矩阵、模型权重与逐样本预测未随已发布 LDA 提交上传。原始数据与衍生材料的分享范围、贡献署名及许可证仍需团队确认。

## 环境记录

本轮新 MLP 使用 Python 3.13.17、PyTorch 2.15.0+cpu、NumPy 2.5.5、scikit-learn 1.9.1；A36 恢复使用 SciPy 1.18.1、h5py 3.16.0 与 FOOOF 1.1.0。精确配置和环境随每次运行冻结保存。

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
| 2026-11-18 | 发布前 13 项测试与暂存区检查通过，将 LDA 更新推送为 `5ce986c`，随后文档更新为 `b72bcb6` | 发布内容不包含原始数据或逐样本运行结果 |
| 2026-11-19 | 恢复全部11,304条A36、冻结内层角色，完成四任务144次固定槽位MLP消融、总览和当前31项测试 | 同会话开发证据；新代码与结果未推送，未恢复旧705维模型权重 |

初始迁移的详细证据仍保存在 `local/source_manifest.json`、`handoff_verification.json`、`migration_regression.json` 等本地文件中。源码对应关系见[资产清单](code_inventory.csv)。

## 复查入口

从仓库根目录执行；以下是复查入口；当前已执行范围以上方记录为准，重新运行仍应选择新的输出目录。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/lda/summarize_lda_results.py --input-dir artifacts/lda/runs/ec --output-dir artifacts/lda/reproductions/validation-check/ec/report
```

选择新的输出目录。训练与图表的完整复现命令见 [LDA Experiment Report](../LDA_EXPERIMENT_REPORT.md)；模型交接状态与下一步大纲见[团队协作说明](08_team_model_integration.md)。

本次后续验收：37项测试通过；48次同角色LDA与全量诊断完成。既有历史节点中的31项指当时检查范围，不是当前测试总数。
