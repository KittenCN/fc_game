# FC Emulator Toolkit

FC Emulator Toolkit 是一个围绕 [`nes-py`](https://github.com/Kautenja/nes-py) 构建的 NES 游戏强化学习与手动游玩工具箱。项目提供从原始 `.nes` ROM 到 Gymnasium 环境、策略网络、训练脚本、日志分析的一站式支持，帮助你快速验证各类强化学习算法（PPO/A2C/RecurrentPPO 等）在经典红白机游戏上的表现。

## 核心特性
- **即插即用的 NES 环境**：封装 `NESGymEnv`，支持 RAM、灰度、RGB 及图像+RAM 组合观测，内置自动按 START 跳过开场动画。
- **丰富的动作与探索策略**：提供多套离散动作模板、热点驱动的宏动作探索、ICM 内在奖励等机制，降低训练早期停滞概率。
- **策略网络预设**：内建多种 CNN / MultiInput / LSTM 结构（`POLICY_PRESETS`），可根据算力快速切换。
- **训练流程完整**：`train.py`/`infer.py`/`cli.py` 覆盖训练、推理、手动游玩，支持 TensorBoard、断点续训、模型按频率自动保存。
- **训练日志分析**：`fc_emulator.analysis` 可统计热点路段、停滞原因、负回合比例、内在奖励等信息，协助定位训练瓶颈。

## 安装
项目默认依赖 Python ≥ 3.10。

```bash
pip install -e .           # 基础功能（手动游玩）
pip install -e .[rl]      # 强化学习相关依赖（Stable-Baselines3、Gymnasium 等）
# 如需图像缩放，可按需安装 pillow 或 opencv-python
```

建议在虚拟环境中安装，并确认显卡驱动已满足 PyTorch 对 CUDA 的要求。

## 快速上手
### 手动游玩
```bash
python -m fc_emulator.cli --rom roms/SuperMarioBros.nes
```
默认按键：`WASD`（方向）、`J`（A）、`K`（B）、`Enter`（Start）、`Right Shift`（Select）。

### 训练示例（PPO）
```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
    --algo ppo --total-timesteps 1000000 --tensorboard \
    --num-envs 6 --vec-env subproc --frame-skip 4 --frame-stack 4 \
    --reward-profile smb_progress --observation-type gray
```
常用参数说明：
- `--observation-type`：`rgb` / `gray` / `ram` / `rgb_ram` / `gray_ram`。
- `--resize H W`：在送入策略前对图像降采样，例如 `--resize 84 84`。
- `--action-set`：选择预设动作集（`default`、`simple`、`smb_forward`）或自定义组合，如 `"RIGHT;A,RIGHT;B"`。
- `--exploration-epsilon` 与 `--entropy-*`：搭配自适应回调，控制探索强度、熵系数衰减。
- `--reward-profile`：选择奖励塑形策略，默认 `none`，可使用 `smb_progress`、`smb_dense`。
- `--icm`：启用 ICM 内在奖励，需使用像素观测。

### 训练断点与推理
- 训练完成后会在 `runs/` 下保存 `ppo_agent_时间戳.zip`。
- 再次运行 `train.py` 时会自动加载最新断点。
- 推理示例：

```bash
python -m fc_emulator.infer --rom roms/SuperMarioBros.nes \
    --model runs/ppo_agent_XXXXXX.zip --observation-type gray \
    --frame-stack 4 --episodes 3 --deterministic
```

## 日志分析与调试
训练脚本可通过 `EpisodeLogCallback` 写入 JSONL 日志（默认 `runs/episode_log.jsonl`）。分析工具示例：

```bash
python -m fc_emulator.analysis runs/episode_log.jsonl --bucket-size 32 --top 10
```
输出包含：
- 平均/中位数 `mario_x` 前进距离与最大值，识别训练进展。
- 负回合占比、内在奖励均值，评估探索质量。
- 停滞（stagnation）发生次数及平均帧数。
- 常见终止原因（如 `stagnation`、`time_limit`）。
- 前进热点区间，定位难点路段。

这些统计有助于判断是否需要调整奖励函数、探索参数或宏动作序列。

## 项目结构
```
fc_emulator/
├── rl_env.py          # Gymnasium 兼容的 NES 环境封装
├── wrappers.py        # 动作包装、探索与图像处理辅助
├── rl_utils.py        # 向量环境工厂、策略映射、动作预设
├── policies.py        # 特征提取器与策略预设
├── rewards.py         # Super Mario Bros 奖励塑形
├── callbacks.py       # 训练日志、探索/熵衰减回调
├── icm.py             # Intrinsic Curiosity Module 包装
├── train.py / infer.py
└── cli.py             # 手动游玩入口
```
示例脚本位于 `examples/`，可用于验证环境与控制器。

## 开发规划（节选）
- 增强 ICM 超参数搜索与可视化，降低探索停滞。
- 丰富状态保存/恢复能力与训练监控指标。
- 探索 LSTM / Transformer 策略及分阶段课程学习。
- 继续完善 README、教程与调参策略文档。

欢迎提交 Issue/PR，共同完善 FC Emulator Toolkit！
