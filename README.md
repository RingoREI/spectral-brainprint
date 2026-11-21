# Spectral Brainprint

**融合周期、非周期与空间统计特征的可解释EEG神经指纹识别框架。**

已有空间协方差和频带功率时，再显式估计 EEG 非周期谱的偏移量与指数，能否提高身份识别准确率？当前实验覆盖 20 个注册身份、四项任务和一个有标签会话。每项任务分别训练和评估。后续研究将检验谱参数与深度表示结合后，增益能否跨记录保持，以及它是否值得模型增加的复杂度。

## 消融实验结果

**统一功率谱密度（PSD）估计后，A36 在四项任务上的 MLP 平均增益都为正，只有闭眼（EC）的三个种子全部提升。** EC 的增益为 **+0.849 个百分点**，EO、Imagery 和 SSVEP 分别为 +0.080、+0.086 和 +0.258 个百分点。原谱方案下四项任务均为 3/3 个种子提升。[英文 Ablation Experiment Report](ABLATION_EXPERIMENT_REPORT.md)

![上图为四个任务在统一 PSD 下四种特征组合的 MLP 平衡准确率，下图逐种子比较两种谱方案下加入 A36 的增益。](docs/results/ablation/ablation-overview.png)

*图 A 中 S 为空间协方差，P 为频带功率，A 为非周期偏移量和指数。图 B 考察 S+P+A 相对 S+P 的平衡准确率差。每条细线连接同一种子在原方案和统一 PSD 下的结果，粗线为三个种子的均值。每个点先合并该种子三折互不重复的外层预测，再计算平衡准确率。* [矢量图](docs/results/ablation/ablation-overview.pdf) · [绘图数据](docs/results/ablation/paired_cells.csv) · [逐种子效应量](docs/results/ablation/paired_effects.csv)

两套谱方案各完成 144 次固定输入维度的 MLP 训练和 48 次同训练划分的 LDA 拟合。统一 PSD 后，LDA 在四项任务上也都获得正的 A36 增益。报告给出了四条件绝对得分、配对差值、种子波动和身份召回下降的实例。[方法与效应量](docs/09_ablation_methods.md) · [来源审计与谱估计验证](docs/10_source_audit_and_matched_psd.md) · [复现命令](scripts/ablation/README.md)

## LDA 实验结果

已完成 EC、EO、Imagery、SSVEP 四任务的三组特征、三折验证，共 **36 次 LDA 训练**。每个任务分别在自己的记录上训练和验证，20 个注册身份均在训练中出现；填补与标准化只在训练折拟合。下面是合并不重复验证预测得到的平衡准确率。

| 任务 | 验证记录 | 功率108 | 协方差171 | 合用279 | 合用相对协方差 |
|---|---:|---:|---:|---:|---:|
| EC | 1,072 | 81.84% | 92.25% | 95.23% | +2.97 pp |
| EO | 1,070 | 76.17% | 87.97% | 91.33% | +3.36 pp |
| Imagery | 5,504 | 92.62% | 95.85% | 97.21% | +1.36 pp |
| SSVEP | 3,658 | 91.96% | 95.97% | 98.27% | +2.31 pp |

![Four-task LDA feature comparison and observed gains](docs/results/lda-within-task-overview.png)

*左图为各任务的合并平衡准确率；右图区分合并预测计算的功率增益与三个实际验证折的增益。逐折点描述当前分块结果，不是置信区间。任务间训练样本量不同。*

四任务、十二个验证折的功率增益方向均为正，逐身份结果也存在回退。当前 LDA 证据支持功率与空间统计在**同会话、任务内开发协议**下的互补性。这项历史 LDA 基线使用较大的训练集。上方消融实验另外按 MLP 的训练划分拟合 LDA，供两种分类器直接比较。

英文 [LDA Experiment Report](LDA_EXPERIMENT_REPORT.md) 统一记录实验设置、全部指标、逐折变化、配对错误、逐身份分析、图表、来源与复现命令。CSV 和各任务图表保存在 `docs/results/`，本地逐次结果集中在被 Git 忽略的目录：

```text
artifacts/lda/
  runs/{ec,eo,imagery,ssvep}/       每任务九个 JSON 与逐折汇总
  reports/{ec,eo,imagery,ssvep}/    各任务表格、PNG/PDF 和 provenance
  overview/                       四任务指标与总览图
  learning/                       早期 EC 第 0 折手工练习
  demo/                           合成示例，验证工程流程
  README.md                       本地目录索引
  migration_manifest.json         已移动文件的原路径、新路径与 SHA256
```

复现时用 `scripts/lda/run_lda_ablation.py` 的 `--task` 选择 `EC`、`EO`、`Imagery` 或 `SSVEP`（默认 EC），再用 `scripts/lda/summarize_lda_results.py` 汇总。每次选择新的输出目录；完整命令见报告。移动文件保留了全部原始内容和来源清单，运行产物继续由 `.gitignore` 排除。

## 当前阶段与下一步

来源审计已将全部 11,304 条记录与交付文件索引、特征和划分对应。真实 Run 和窗口在原始连续记录中的位置仍未知。下一步是取得切片来源映射，或取得相同注册身份的另一段有标签采集，以检验当前增益能否跨记录保持。[审计方法和结果](docs/10_source_audit_and_matched_psd.md)

消融使用重新训练的 315 维 MLP。旧 705 维 MLP20 的权重与金样仍待交接，原始 EEG 的 CNN 融合和 GNN 仍待实验验证。[团队协作说明](docs/08_team_model_integration.md)

## 从这里开始

- [项目定位与证据边界](docs/01_positioning.md)：研究目标、交付对象、应用边界。
- [转化路线与验收条件](docs/02_roadmap.md)：近期行动、阶段依赖与完成标准。
- [仓库架构与迁移规则](docs/03_repository.md)：目录职责、代码去向与发布规则。
- [已有资产审计](docs/04_asset_audit.md)及[逐文件清单](docs/code_inventory.csv)：本地材料实际支持什么。
- [数据与模型契约](docs/05_contracts.md)：给非周期分支、学长的统一接口。
- [实验协议](docs/06_experiments.md)：基线、增量比较、拒识与抗扰。
- [验证状态与历史记录](docs/07_validation.md)：已完成检查、当前证据范围、不同阶段的环境与未验证事项。
- [团队协作、模型状态与下一步实验](docs/08_team_model_integration.md)：现有模型能力、待交接资产、建议分工、当前消融进展与后续大纲。
- [2×2 消融方法与效应量](docs/09_ablation_methods.md)：设计矩阵、百分点差、条件效应、交互项、补充指标和不确定性报告规则及执行前提。

## 本地起步

使用独立环境，最低 Python 3.10。四任务 LDA 的实际环境见[验证状态](docs/07_validation.md)与各任务来源清单；`environments/validated-current.txt` 保留的是 11 月 12 日初始迁移环境，`environments/legacy-requirements.txt` 是原交接环境记录，两者均不代表已恢复完整团队训练环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m spectral_brainprint demo --output artifacts/lda/demo/demo-rerun.json
```

模拟流程只验证工程连通性，输出不用于证明真实脑纹准确率。读取已有交接包时，显式提供 `--package-root`，该值为包含 `handoff_manifest.json` 与 `payload/` 的目录。先复现一个案例：

```powershell
python -m spectral_brainprint reproduce --package-root "<交接包目录>" --feature-set power108 --protocol within_task_blocked_provisional --task EC --fold 0 --output artifacts/lda/learning/ec-fold0-rerun.json
```

此入口只用已有Train特征，校验索引/列定义、检查训练验证记录和身份，再在训练折拟合填补、标准化与LDA，并与历史同一案例的指标比较。它不训练最终团队模型，也不校准拒识阈值。

## 当前证据

本地交接包包含11,304条Train、22,251条无身份真值Test，四任务，18通道、300 Hz、每段4秒。Train均为session1；文件编号分块未经确认对应真实Run/时间。历史28案例属于探索性同会话协议，不能写成跨会话验证、官方攻击成绩或成熟身份认证系统。

默认输入保持 `data_basic`、原参考、未核实原生幅值单位。带通功率包含周期和非周期成分；宽带协方差是混合统计。可解释特征不自动意味着纯脑源、因果解释或排除设备/伪迹混杂。

## 资料管理

仓库中包含代码、配置、合成示例、英文实验报告及按任务汇总的主图和数值表。原始 MAT、H5、NPZ 保持在原资料目录，通过显式路径读取；`artifacts/` 中的逐样本结果、`local/` 中的源码快照与核验材料、模型权重和虚拟环境继续由 Git 忽略。贡献署名、数据与衍生材料的分享范围及许可证仍需团队确认。
