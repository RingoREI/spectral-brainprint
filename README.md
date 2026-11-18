# Spectral Brainprint

**融合周期、非周期与空间统计特征的可解释EEG神经指纹识别框架。**

当前是从竞赛脚本向可复现研究软件迁移的早期仓库。已迁入功率、空间与盲谱计算、交接包读取和对齐、固定 LDA 单案例复现、团队 MLP20 的融合特征读取与冻结阈值推理。融合网络、ResMLP 和原始波形 CNN 已迁入模型定义。真实 MLP20 推理需要交接版对应的权重和校准文件。非周期波形提取与图模型属于后续接入范围。

项目首先回答：非周期参数是否在功率和空间基线之外提供身份信息，以及这种增益能否跨任务、记录组和扰动条件保持。深度阶段描述为“结合谱参数化先验与深度表示学习的EEG神经指纹识别算法”；收益与复杂度均须通过实验验证。

## 已复现的 LDA 特征对照

[Experiment Note 01](brainprint-experiment-note.md) provides the experimental setup, pooled and per-fold results, error analysis, and limitations in English.

闭眼任务 EC 的同会话开发实验使用 20 个注册身份、三折文件编号分块划分。每组特征使用同一批训练和验证记录，填补与标准化仅在训练折拟合。下面的总体指标由三折共 1,072 条不重复的验证预测合并计算。

| 特征 | 维数 | 准确率 | 平衡准确率 | Macro-F1 |
|---|---:|---:|---:|---:|
| 频带功率 | 108 | 82.46% | 81.84% | 82.18% |
| 对数协方差 | 171 | 92.54% | 92.25% | 92.66% |
| 功率＋对数协方差 | 279 | **95.62%** | **95.23%** | **95.55%** |

功率加入协方差基线后，合并平衡准确率增加 **2.97 个百分点**，三个折的增益依次为 **4.58、1.64、2.68 个百分点**。在相同验证记录上，40 条预测由错误变为正确，7 条由正确变为错误。该对照支持功率与空间统计在当前开发协议下的互补性。

![三组 LDA 的总体指标与逐折平衡准确率](docs/results/lda-ec-within-task/lda_performance.png)

*左图为合并验证预测的指标，右图展示三组特征各自在三个验证折中的表现。折间变化来自文件编号分块划分。当前结果覆盖一个有标签会话，跨会话和未知身份表现仍待独立评估。*

逐身份召回率显示 16 个身份改善、3 个持平、1 个下降。身份 05 从协方差基线的 95.0% 降至 87.5%，应和整体增益一并检查。详细图表、混淆矩阵、CSV 和 PDF 见[实验报告](docs/results/lda-ec-within-task/README.md)。

安装绘图依赖后，用已有逐条预测生成一份新的报告目录。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
.\.venv\Scripts\python.exe scripts/summarize_lda_results.py --input-dir artifacts/lda-ec-all-folds --output-dir artifacts/lda-ec-report
```

`--skip-plots` 只导出表格。每次选择新的输出目录，保留已生成的报告。汇总脚本会检查记录对齐、跨折验证记录无重复、输入哈希一致，并从逐条预测重新计算指标。原始逐条预测保存在被 Git 忽略的 `artifacts/` 中，README 图表使用汇总结果。

## 从这里开始

- [项目定位与证据边界](docs/01_positioning.md)：研究目标、交付对象、应用边界。
- [转化路线与验收条件](docs/02_roadmap.md)：近期行动、阶段依赖与完成标准。
- [仓库架构与迁移规则](docs/03_repository.md)：目录职责、代码去向与发布规则。
- [已有资产审计](docs/04_asset_audit.md)及[逐文件清单](docs/code_inventory.csv)：本地材料实际支持什么。
- [数据与模型契约](docs/05_contracts.md)：给非周期分支、学长的统一接口。
- [实验协议](docs/06_experiments.md)：基线、增量比较、拒识与抗扰。
- [本次验证记录](docs/07_validation.md)：本次实际执行情况，不等同于历史报告。
- [团队模型接入和下一轮实验](docs/08_team_model_integration.md)包含 MLP20 运行命令、功率与非周期消融设计，以及图模型和 GitHub 发布顺序。

## 本地起步

使用独立环境。最低Python 3.10；已验证版本见 `environments/validated-current.txt`。`environments/legacy-requirements.txt` 是原交接包环境记录，不代表本机已经恢复该环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m spectral_brainprint demo --output artifacts/demo.json
```

模拟流程只验证工程连通性，输出不用于证明真实脑纹准确率。读取已有交接包时，显式提供 `--package-root`，该值为包含 `handoff_manifest.json` 与 `payload/` 的目录。先复现一个案例：

```powershell
python -m spectral_brainprint reproduce --package-root "<交接包目录>" --feature-set power108 --protocol within_task_blocked_provisional --task EC --fold 0 --output artifacts/ec_fold0.json
```

此入口只用已有Train特征，校验索引/列定义、检查训练验证记录和身份，再在训练折拟合填补、标准化与LDA，并与历史同一案例的指标比较。它不训练最终团队模型，也不校准拒识阈值。

## 当前证据

本地交接包包含11,304条Train、22,251条无身份真值Test，四任务，18通道、300 Hz、每段4秒。Train均为session1；文件编号分块未经确认对应真实Run/时间。历史28案例属于探索性同会话协议，不能写成跨会话验证、官方攻击成绩或成熟身份认证系统。

默认输入保持 `data_basic`、原参考、未核实原生幅值单位。带通功率包含周期和非周期成分；宽带协方差是混合统计。可解释特征不自动意味着纯脑源、因果解释或排除设备/伪迹混杂。

## 资料管理

`local/` 保存本次本地源码快照和核验记录，已被Git忽略；其中历史脚本可能含原机路径。原始MAT、H5和NPZ保持在原资料目录，通过显式路径读取。发布候选仅包含已整理的包、配置、合成示例和文档。当前没有选定开源许可证或声明原代码作者；应在贡献归属和数据授权核实后决定。不要把包含 `local/` 的整个本地工作目录直接公开。
