# 合成示例

从仓库根运行 `python -m spectral_brainprint demo --output artifacts/demo.json`。
示例内置于CLI，生成36条合成18通道记录，计算108维功率与171维对数协方差，按独立生成的记录划分训练和验证，运行LDA。
这是安装和工程连通性检查，合成类别人为包含差异，不代表真实EEG身份可识别性。

现有单记录H5查看脚本保留在本地legacy快照；参数化版本尚未迁入。
