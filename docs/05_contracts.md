# 数据、特征与模型契约 v0.1

本契约是团队接入目标。现有 `legacy_handoff.py` 保留旧NPZ读取接口；未来分支按本契约交付并经适配器进入核心。

## 记录与输入

每条记录具有 `record_id`、`path`、`split`、`task`、`session`；Train具 `identity_label`，Test无真值。`record_id` 只有在同一索引命名空间内才可比较；重编号分支通过完整规范path或已核实信号哈希建立映射。标准path是索引内原记录路径，不是当前磁盘根路径。

原始数组约定 `[N, time=1200, channel=18]`；网络若用 `[N,C,T]` 由适配器显式转置。采样率300 Hz，通道顺序和单位见配置。每个分支声明输入视图 `base/clean/removed/ica`；不得把不同视图无说明拼接，ICA同序号成分不能视为跨记录同源。

未来增加 `source_record_group`、`run_id`、`window_start`、`parent_record_id` 和 `augmentation_id`。未知时为缺失，不能用猜出的编号制造独立组。所有原件/增强/重叠切片必须留在同一划分。

## 特征交付

| 字段 | 约定 |
|---|---|
| X | N×D数值矩阵；允许有明确语义的NaN；禁止Inf和混入标签/ID/路径/任务/会话 |
| record_ids / paths | 至少一个已验证的标准主键，建议两者；一对一且不静默删行 |
| feature_names | D个唯一列名；Train/Test严格同顺序 |
| identity_labels | 仅Train提供并核对；Test不填假标签0 |
| quality | 独立表，以record_id关联；默认不拼入身份X |
| feature_dictionary | 列名、通道/频带、单位、变换、范围、缺失含义、版本 |
| provenance | 输入索引/视图、配置SHA256、依赖版本、提取代码版本 |

质量状态至少区分 `ok`、`no_peak`、`fit_failed`、`low_quality`、`invalid_input`。无峰并不意味着非周期拟合失败；低质量与是否存在峰也可并存，宜按分支/通道分别记录状态。连续峰参数可为NaN，若添加峰存在指示，应作为明确的峰特征消融；拟合误差、R²、失败状态等质量指标默认只用于诊断。填补和缺失指示仅在训练折拟合，并单独检查它们是否构成质量混杂。

公共块去重：P108一次、S171一次；P+S=279，附加盲谱144后=423。A18/A36列定义由实际分支确认，offset与exponent使用的log底/PSD单位必须说明。矩阵log是自然对数，功率是log10，二者不得互换。

## 模型包

模型交付目录至少包含：`model_config.json`、训练/推理源码、权重、预处理器、`feature_dictionary.csv`、`class_mapping.json`、`split_manifest.csv`、依赖锁定记录、训练日志、`metrics.json` 和 `golden_input`/`golden_output`。可使用有授权的匿名或合成金样。

预处理器包括填补、缩放、选择器/PCA（若用）及其训练范围；网络内部标准化、batch norm统计和推理模式也应说明。列索引、类别索引与身份编码分离：softmax第0列不自动代表身份0。`predict`返回身份；`predict_proba`需固定class_order；`encode`若存在则说明embedding维数与归一化。

拒识另交 `rejection_config.json`：score的定义、数值方向、阈值、不等式及等号处理、校准数据与目标。建议输出 `predicted_identity`、`score`、`accepted`、`rejection_reason`；仅在竞赛输出适配层把拒绝映射为0。坏输入应返回明确错误/状态，不能混同于未知身份。

## 最小接收验收

恢复乱序结果不变；重复/缺失/额外键或标签冲突立即报错；训练/Test字典匹配；金样在指定环境可复现；所有学习型变换仅用允许的训练/开发数据；无身份泄漏和增强跨分区；固定阈值后才运行最终评估。完整model包接收前，不对接深度融合头。
