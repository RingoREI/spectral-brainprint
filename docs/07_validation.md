# 本次验证记录

日期：2026-11-12。以下均为本次执行结果；原始数据和交接包只读，核验脚本的报告写入重定向至新仓库 `local/`。

| 检查 | 结果 | 证据与范围 |
|---|---|---|
| 源码冻结 | 20个Python文件原样复制并记录SHA256 | `docs/code_inventory.csv`，`local/source_manifest.json`；未复制全量EEG |
| 交接包完整性 | 48文件哈希/大小全部匹配 | `local/handoff_verification.json`；原清单内文件 |
| Train/Test对齐 | 通过 | 279维全量主键/路径/任务/会话；Train标签及Test模板行序 |
| V4九块 | 通过 | 全Train/Test维数与有限值；共1305列 |
| 28案例 | 通过 | 每案例训练验证ID无交集，身份集合符合原20类闭集；不证明真实来源组无重叠 |
| 旧报告一致性 | 通过 | V4与V2的power108历史指标相等；这是报告对照，不是全表重训练 |
| 迁移数值回归 | 最大绝对差0 | 3条合成+3条真实记录；PSD、功率及全部九块；新旧函数在同一当前环境下运行 |
| 关键单元测试 | 5项通过 | 乱序恢复与冲突、显式路径、记录泄漏/闭集条件、训练内变换、功率尺度/OAS性质 |
| 模拟端到端 | 通过 | 36条合成记录→279维→LDA，24训练/12验证；合成准确率不作真实脑纹证据 |
| 一个真实历史案例重训练 | 三项指标差均为0 | power108 / within_task_blocked_provisional / EC / fold0；670训练，362验证 |
| 打包与安装 | 通过 | 离线构建wheel，在单独venv安装并以Python隔离模式从仓库外运行模拟CLI；复用系统已有依赖，不等同于全新依赖环境 |

打包时首次遇到本机沙箱默认pip缓存路径不可用，改用工作目录内短临时与缓存路径后成功；无需改算法或联网安装。已初始化本地main分支，无提交、无远程。默认候选跟踪目录中未检出私人绝对路径，`local/`已验证被Git忽略。

本次EC单折结果：accuracy=0.7817679558011049，balanced accuracy=0.7702119268760135，macro-F1=0.7483835015696148，与原逐折表一致。不要将此值替代历史EC全部折合并结果。逐记录预测保存在本地 `local/ec_fold0.json`。

数值回归命令：

```powershell
python scripts/check_migration.py --source-root "<原myBrainPrintProject目录>" --output artifacts/migration_regression.json
```

当前Python3.10.10、NumPy2.2.6、SciPy1.14.1、scikit-learn1.7.2、h5py3.16.0、threadpoolctl3.6.0。历史requirements含不同版本；本次验证不等同于恢复历史环境。

完整28案例×16表示尚未重训练；未重跑全量特征提取；未恢复非周期/团队MLP、校准拒识、训练深度融合或验证真实攻击/跨会话。全新无共享依赖环境的安装属于公开发布前验收。
