# 仓库架构与迁移规则

建议项目目录继续容纳现有数据资料，在其中新增独立算法仓库 `spectral-brainprint/`。现有 `Brainprint_Competition/` 与 `Processed Data/` 保留来源含义；不要把数据根目录整体初始化成公共Git仓库。

```text
myBrainPrintProject/
  Brainprint_Competition/       原始材料，保持位置
  Processed Data/               历史处理/交接结果，保持位置
  Brainprint_ML/                团队深度模型交接版与历史实验入口
  spectral-brainprint/
    pyproject.toml
    README.md
    configs/{data,psd,features,splits,models}/
    src/spectral_brainprint/
      io/                      索引、H5/NPZ适配、严格对齐
      spectra/                 Welch与后续谱参数化
      features/                功率、空间、盲谱；非周期待接入
      models/                  LDA、MLP20 推理、融合网络、ResMLP 与波形 CNN
      evaluation/              案例读取与复现；拒识/分组协议待扩展
    examples/                  纯合成示例入口及说明
    tests/                     数值性质、对齐冲突、泄漏约束
    docs/                      定位、协议、契约和资产证据
    scripts/                   本地迁移回归等维护脚本
    environments/              历史环境记录、当前已验证版本
    artifacts/                 本地生成结果，Git忽略
    local/                     本地源码快照、私有配置/核验，Git忽略
```

## 模块边界

IO负责读取、身份或主键核对及保持记录顺序。谱和特征模块做不依赖身份标签的逐记录变换。模型管线负责训练折内的填补、缩放与拟合。`io/team_vectors.py` 提供历史 MLP20 向量化及归一化规则，调用方显式传入训练行或冻结参数。`evaluation/` 负责划分、指标和模型选择边界。谱参数化输出特征与质量诊断两个对象。

目前迁移的是可明确识别的纯数值部分。`features/spatial_blind.py` 暂保留旧V4计算和字典，后续数值回归完成后再拆为空间与盲谱两个文件。`io/legacy_handoff.py` 是版本特定适配器，不宣称是通用数据加载器。所有实际数据路径通过参数提供，禁止导入模块时读取私人文件或启动绘图。

## 旧代码处理

完整列表见 `code_inventory.csv`；20个源文件以原相对层级保存在 `local/legacy_snapshot/`，并记录SHA256。快照不参与默认导入/安装，也不代表各脚本均可独立运行。原脚本及资料未移动、删除或覆盖。

`basic_analysis.py` 是入门单记录查看脚本，有硬编码路径、标题及顶层绘图，先冻结；下一次改为带参数的 `examples/inspect_record.py`。历史前处理的通配导入、路径绑定及按身份ICA不能直接混入新评估链路；逐记录ICA也应作为独立实验视图。V2/V3的报告并不等于对应峰提取源码已齐备。

## 配置与结果版本

- 保留原始V1/V4配置原字节；新增PSD或拟合方式另建配置版本，不能悄悄改名覆盖。
- 公共配置不填私人根路径。路径采用CLI或未跟踪的 `local/paths.json`；先确认存在再读。
- 每次实验输出代码版本/哈希、环境、配置哈希、特征字典哈希、划分哈希、记录数量、随机种子、指标和逐条预测。当前最小入口记录主要输入哈希与环境；完整实验管理尚待G1扩展。
- 结果放 `artifacts/<run_id>/`，每次独立；同路径存在时拒绝覆盖。不要在源码或冻结数据目录直接生成新结果。

## 版本管理策略

本次准备独立本地Git仓库，不创建远程、不推送、不代填作者或提交署名。首次提交前检查暂存列表，仅纳入代码、公开配置、合成示例与说明。暂不生成LICENSE或臆造CITATION作者；贡献归属和共享边界在学长交接时确认。

`local/`、数据缓存、模型权重、虚拟环境、图形缓存和运行产物均忽略。忽略规则只是操作辅助，公开前仍需检查跟踪文件和发布包。README不能宣称深度模型、公开数据和完整复现已具备。
