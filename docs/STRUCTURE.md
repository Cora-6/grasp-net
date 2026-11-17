# 代码阅读指南

项目的主线是：读取同一批 RGB-D 样本，用不同网络预测抓取图，再用同一指标比较正常和退化输入。
代码按职责组织；RQ1/RQ2/RQ3 共用数据和评测，差别由实验配置表达。

## 推荐阅读顺序

| 顺序 | 文件 | 先看什么 |
|---|---|---|
| 1 | `configs/comparison.json` | 比较哪些模型、哪些种子、训练多少轮 |
| 2 | `datasets/graspnet.py` | RGB、深度、矩形标注的读取与缩放 |
| 3 | `datasets/base.py` | 旋转、亮度增强、退化、归一化和四通道输入 |
| 4 | `datasets/targets.py` | 矩形标注如何生成质量、角度、宽度标签 |
| 5 | `models/registry.py` | 实验名称与实际网络的对应关系 |
| 6 | `models/fusion.py` | 双分支和近等容量单分支的区别 |
| 7 | `losses.py` | 标签协议转换、原始损失与前景平衡损失 |
| 8 | `engine/trainer.py` | 一个 epoch、验证选模与训练恢复 |
| 9 | `evaluation/decoder.py`、`evaluation/geometry.py` | 四张预测图如何变成矩形、如何判断匹配 |
| 10 | `evaluation/evaluator.py` | 固定退化条件、逐图记录、指标和图表 |

`engine/cli.py` 负责参数解析和训练/评测分发。`train.py` 与 `evaluate.py` 是简短入口。
`scripts/run_model_comparison.py` 只组织多模型任务；实验算法放在上述模块中。

## 容易混淆的名称

| 名称 | 含义 |
|---|---|
| `ggcnn` / `ggcnn2` | 本项目中的 RGB-D 适配版本；模型标识沿用旧实验 |
| `ggcnn_depth` / `ggcnn2_depth` | 只把深度通道传入网络 |
| `GGCNNAdapter` / `GGCNN2Adapter` | 同时支持 RGB-D 和 Depth 的适配器，因此不再用 RGBD 命名整个类 |
| `early_concat` | 输入阶段拼接 RGB 和深度，共用编码器 |
| `dual_concat` | RGB 和深度分别编码，再拼接特征 |
| `early_fusion` | `DualGraspNet` 内选择单分支结构的开关 |
| `use_gate` | 特征融合的门控开关；当前主实验的两种融合模型均关闭 |
| `baseline` | 不额外施加传感器退化；训练仍有旋转和亮度增强 |
| `audit/` | 数据检查的输出目录，不是训练模块，也不提交到 Git |

内部样本通道顺序保持 R/G/B/D；上游模型所需的输入排列由对应适配器处理。
数据层保留四通道形式，使 Depth-only 和 RGB-D 实验使用一致的样本与退化条件。

## 配置与结果

单模型 JSON 对应 `train.py` / `evaluate.py` 的参数；`comparison.json` 对应批量对照脚本，
两类配置不能互换。字段使用下划线，例如 `loss_profile` 对应 `--loss-profile`。
未知字段、错误的类型与不合法的选项会报错。命令行同名参数优先；布尔开关设为 true 后，
若要关闭，应修改配置文件为 false。相对路径以运行命令时的工作目录为基准。

单模型配置明确指定 300 像素、20 轮、batch 8、学习率 0.001、原始损失和正常验证集选模。
完整对照配置使用相同协议，覆盖七个模型和三个种子。
新增实验时复制一份配置并更改输出路径，可避免覆盖已有结果。

| 文件 | 用途 |
|---|---|
| 输入 `configs/*.json` | 可重复使用的实验预设 |
| 输出 `config.json` | 实际参数、模型协议、版本和 split 哈希 |
| `history.json` / `progress.json` | 已完成轮次 / 当前批次进度 |
| `best.pt` / `last.pt` | 验证集最优权重 / 最近完整 epoch 的恢复状态 |
| `comparison_protocol.json` | 对照模型、种子与数据协议 |
| `test_*/metrics.csv` | 各退化条件的匹配率 |
| `test_*/predictions.json` | 逐样本结果 |
| `test_*/evaluation_manifest.json` | 测试集、权重哈希、随机种子等来源 |

## 兼容与验证

旧的 `run.py train/evaluate` 命令、`dataset.GraspNet`、`models` 公共导入和
`GGCNNRGBD` / `GGCNN2RGBD` 类名仍保留。新代码推荐直接导入职责明确的模块。
`grasp.py`、`dataset.py`、`augmentations.py` 现在只转发旧接口，避免维护两份实现。
历史 Cornell 和 Tiny U-Net 实现放在 `legacy/`；GraspNet 使用独立命名的数据类。
模型参数键保持不变，已有 state_dict 权重仍可加载。

运行 `python -m unittest discover -s tests -v` 检查数据处理、模型损失、配置覆盖，
并用临时合成样本验证训练、断点续训、旧入口以及全部 11 种评测条件。
续训测试对比连续训练和恢复训练的最终权重，要求逐元素完全一致。
这些小样本测试用于检查实现，不代表实验准确率；正式结果仍需使用完整划分。

进度服务位于 `dashboard/server.py`，由 `scripts/modality_dashboard.py` 启动。
根目录的 `progress_server.py` 和 `main_training_dashboard.py` 是历史页面，保留供旧流程使用。
