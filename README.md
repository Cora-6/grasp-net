# RGB-D Grasp Detection: GraspNet Experiments

本项目比较 GG-CNN、GG-CNN2、GR-ConvNet 与双分支抓取检测网络，
并通过近等容量单分支研究 RGB 与深度分别编码的作用。

## 模型

| CLI 名称 | 输入 | 说明 |
|---|---|---|
| ggcnn_depth | Depth | GG-CNN 单通道 |
| ggcnn | RGB-D | GG-CNN 四通道适配 |
| ggcnn2_depth | Depth | GG-CNN2 单通道 |
| ggcnn2 | RGB-D | GG-CNN2 四通道适配 |
| grconvnet | RGB-D | GR-ConvNet |
| dual_concat | RGB-D | 分别编码后拼接融合，use_gate=False |
| early_concat | RGB-D | 近等容量单分支，early_fusion=True |

`models/fusion.py` 另有门控模块，但当前主实验不启用。
`original` 损失配置对 GG-CNN/GG-CNN2 使用四输出 MSE，对其余三个模型使用四输出 Smooth-L1。
`foreground_balanced` 是项目提供的另一种损失配置。架构适配不等于完整复现原论文训练方案。

## 安装与检查

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

`requirements-lock.txt` 保存原环境版本，跨平台安装建议先使用 `requirements.txt`。

## 数据

使用 GraspNet-1Billion RealSense 抽帧数据，帧间隔 16；数据不包含在仓库中。
`split.json` 固定包含训练 1,440、验证 160、已见/相似/新物体测试各 480 帧。
验证集是项目设置的十个训练场景，测试组沿用数据集场景划分。

文件应位于：

```text
data/scenes/scene_0000/realsense/rgb/0000.png
data/scenes/scene_0000/realsense/depth/0000.png
data/scenes/scene_0000/realsense/rect/0000.npy
```

所有清单帧都需要 RGB、深度和二维矩形标注。实验入口在缺失文件时会停止。
下载工具 `scripts/download_data.py` 使用第三方公开镜像；下载是否包含全部测试矩形标注需实际检查。
`scripts/fetch_generation_inputs.py` 只获取补充标注所需输入，不负责完整生成二维标注。
遵守原数据集和镜像的许可，勿把数据重新提交到仓库。

## 运行实验

以下命令在仓库根目录运行。配置文件集中在 `configs/`，命令行参数覆盖配置中的同名字段；
路径相对于当前工作目录。更改数据位置时加 `--data /path/to/data`。

先运行单模型训练：

```bash
python train.py --config configs/ggcnn2_depth.json
```

评测选出的权重：

```bash
python evaluate.py --config configs/ggcnn2_depth.json \
  --checkpoint outputs/ggcnn2_depth_s42/best.pt \
  --part test_seen --out outputs/ggcnn2_depth_s42/test_seen --no-visuals
```

`configs/` 中七个模型各有一份单模型配置，使用同一训练协议。
单模型训练只做训练和验证；要自动训练并评测三个测试集，使用对照入口：

```bash
python scripts/run_model_comparison.py \
  --data ./data --split ./split.json --out ./outputs/ggcnn2_depth_run \
  --research-plan --models ggcnn2_depth --seeds 42 --epochs 20 --device cpu
```

运行完整模型对照：

```bash
python scripts/run_model_comparison.py --config configs/comparison.json
```

统一输入 300×300、batch 8、Adam 0.001；正常验证集选择权重。
`baseline` 不额外加入传感器退化，仍使用数据加载器中的普通训练增强。
对照队列完成训练后评测三个测试集；用 `--resume` 恢复已有输出目录。
更改模型、种子或超参数时应使用新输出目录。
旧命令 `python run.py train ...` 和 `python run.py evaluate ...` 仍可使用。
旧命令默认值保留；推荐使用上述配置文件明确实验协议。

评测包含正常、深度噪声/深度缺失/RGB-D 遮挡各三档以及完全关闭深度，共 11 种条件。
每图保留一个最高分抓取，与任一标注满足角度误差 <30° 且 IoU >0.25 即为正确。
这是离线二维矩形匹配率，不是 GraspNet 官方 6DoF AP，也不是实物抓取成功率。

## 代码和输出

推荐从配置开始，沿着数据 → 模型 → 损失 → 训练 → 评测阅读。

| 位置 | 职责 |
|---|---|
| `configs/` | 单模型与完整对照的实验设置 |
| `datasets/` | GraspNet 读取、RGB-D 预处理、标签生成、退化 |
| `models/registry.py`、`models/*.py` | 模型创建、架构与输入适配 |
| `losses.py` | 训练目标及损失配置 |
| `engine/trainer.py` | 训练、验证选模、保存与恢复 |
| `evaluation/` | 预测解码、矩形匹配、退化评测 |
| `scripts/run_model_comparison.py` | 多模型、多种子任务调度 |
| `dashboard/server.py` | 读取实验记录并提供进度页面 |
| `legacy/` | 历史 Cornell 适配与 Tiny U-Net |

完整阅读顺序、兼容入口和参数解释见 [代码结构说明](docs/STRUCTURE.md)。
`dataset.py`、`grasp.py`、`augmentations.py`、`run.py` 保留为旧接口兼容层。

每个训练目录下的 `config.json`、`history.json`、`best.pt`、`last.pt` 分别保存解析后的设置、
训练记录、最优权重和可恢复训练状态；`test_*/metrics.csv`、`predictions.json`、
`evaluation_manifest.json` 保存测试成绩、逐图记录和来源。

数据集、权重、历史实验输出与本机日志均不包含在代码仓库中。
历史 PPT 的 GR-ConvNet 曾复用不同尺寸/损失配置的权重，不能把历史表格称为完全统一配置对照；
上面的完整命令会从头训练指定模型，生成本次运行自己的结果。

## 本地进度页面

```bash
python scripts/modality_dashboard.py --runs outputs/comparison
```

打开 http://127.0.0.1:18767/ 。每五秒读取实验记录。
`--runs` 支持一个或多个对照目录、单模型训练目录；`--port` 可更换端口。
对照目录根据计划显示尚未启动的任务，单模型目录需已有 `config.json`。
页面展示标准 11 条件评测的进度。

不传 `--runs` 时保留原有三组补充实验路径：
`outputs/ggcnn_modalities/ggcnn_s42`、`outputs/ggcnn_modalities/ggcnn2_depth_s42`、
`outputs/ggcnn2_comparison/ggcnn2_s42`。
页面不会启动训练。`progress_server.py` 与 `main_training_dashboard.py` 为早期实验页面。

## 来源

GG-CNN/GG-CNN2 来自 https://github.com/dougsm/ggcnn ，版本
`0c50aa7600e8a30d44c5c85cebd6e3394a81f30e`。
GR-ConvNet 来源和版本见 `third_party/grconvnet/SOURCE.json`。
第三方许可证保留在各自目录，`SOURCE.json` 用于核对第三方代码完整性。
融合模块含参考 SSMA 的独立实现（https://arxiv.org/abs/1808.03833），当前对照关闭门控。
