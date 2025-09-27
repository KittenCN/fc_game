# FC Emulator Toolkit

FC Emulator Toolkit 基于 [nes-py](https://github.com/Kautenja/nes-py) 搭建，围绕 NES 红白机环境提供从 ROM 加载、Gymnasium 封装、策略库、训练脚本到日志分析的一体化工作流。近期重构聚焦于环境抽象、探索机制、诊断与日志分析，使在经典游戏上的强化学习实验更加稳定、易于复现与调试。

## 项目亮点
- **模块化环境封装**：`fc_emulator.rl_env` 内置 `AutoStartController`、全新的 `StagnationMonitor` 与 `RewardConfig`，支持动态停滞阈值、`score_loop` 检测以及 `stagnation_idle_frames` 指标，环境信息会同步写入 `info.metrics`。
- **动作与探索解耦**：动作模板、图像处理与宏动作探索分别位于 `actions.py`、`observation.py`、`exploration.py`，`wrappers.py` 仅提供兼容入口，便于自定义策略组合。
- **结构化训练配置**：`train.py` 通过 `TrainingConfig` 统一封装环境、算法、探索与日志参数，自动导出 `runs/run_config.json` 以复现实验，新增 CLI 选项可调整诊断频率与停滞策略。
- **在线诊断回调**：`DiagnosticsLoggingCallback` 周期性输出平均前进距离、热点位置、停滞原因比例与内在奖励均值到 TensorBoard（`diagnostics/*`），`EpisodeLogCallback` 则持续落地 JSONL。
- **增强日志分析**：`fc_emulator.analysis` 支持热点分桶、停滞原因分布、近期窗口统计以及 `--json` 输出，可在不加载全部数据的情况下快速审阅大规模日志。

## 目录结构
```
fc_emulator/
├── actions.py              # 离散动作模板与包装器
├── auto_start.py           # 标题画面自动跳过控制
├── exploration.py          # 宏动作与热点驱动探索包装
├── observation.py          # 图像缩放与帧堆叠包装
├── stagnation.py           # 停滞检测、score_loop 识别
├── rl_env.py               # NESGymEnv 主循环
├── rl_utils.py             # 向量化环境与算法映射
├── callbacks.py            # 训练回调（日志、诊断、熵/ε 调度）
├── analysis.py             # JSONL 日志分析工具
├── train.py / infer.py     # 训练与推理入口
└── wrappers.py             # 向后兼容的聚合入口
```

## 安装
要求 Python ≥ 3.10，推荐使用虚拟环境：

```bash
pip install -e .          # 仅运行模拟器 / CLI
pip install -e .[rl]      # 启用强化学习（Stable-Baselines3、Gymnasium 等）
# 若需图像缩放，可额外安装 pillow 或 opencv-python
```

## 快速上手
### 手动游玩
```bash
python -m fc_emulator.cli --rom roms/SuperMarioBros.nes
```
默认按键：`WASD`（方向）、`J`（A）、`K`（B）、`Enter`（Start）、`Right Shift`（Select）。

### 强化学习训练（PPO 示例）
```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
  --algo ppo --total-timesteps 1000000 --num-envs 6 --vec-env subproc \
  --frame-skip 4 --frame-stack 4 --reward-profile smb_progress \
  --observation-type gray --tensorboard --diagnostics-log-interval 2000
```
常用参数：
- `--observation-type`：`rgb` / `gray` / `ram` / `rgb_ram` / `gray_ram`
- `--resize H W`：送入策略前的降采样尺寸（如 `--resize 84 84`）
- `--action-set`：预设动作（`default`、`simple`、`smb_forward`）或自定义组合 `"RIGHT;A,RIGHT;B"`
- `--exploration-*`：配置 epsilon 衰减以及宏动作探索强度
- `--entropy-*`：通过 `EntropyCoefficientCallback` 线性衰减策略熵系数
- `--reward-profile`：`none` / `smb_progress` / `smb_dense`
- `--icm`：启用内在奖励模块（需像素观测）
- `--diagnostics-*`：控制诊断日志频率、窗口长度以及热点分桶

训练流程会自动：
1. 在 `runs/run_config.json` 存储完整配置以便复现。
2. 如存在断点，自动加载最新 `ppo_agent_*.zip`。
3. 将诊断指标写入 TensorBoard，并在 JSONL 中记录每个 episode 的奖励、停滞信息与 `stagnation_idle_frames`。

### 推理示例
```bash
python -m fc_emulator.infer --rom roms/SuperMarioBros.nes \
  --model runs/ppo_agent_XXXXXX.zip --observation-type gray \
  --frame-stack 4 --episodes 3 --deterministic
```

## 日志与分析
- `EpisodeLogCallback` 默认写入 `runs/episode_log.jsonl`，记录基础奖励、塑形奖励、内在奖励、停滞原因以及 `stagnation_idle_frames`。
- `DiagnosticsLoggingCallback` 会把平均 `mario_x`、近期均值、热点位置、停滞原因占比与内在奖励均值写入 TensorBoard（`diagnostics/*`）。
- `fc_emulator.analysis` 提供热点分桶、停滞原因统计、近期窗口指标与 `--json` 输出：
  ```bash
  python -m fc_emulator.analysis runs/episode_log.jsonl --bucket-size 32 --top 10
  python -m fc_emulator.analysis runs/episode_log.jsonl --json
  ```

## 近期问题与解决方案
- **训练频繁被停滞截断**：旧版停滞阈值固定 752 帧，导致中后期探索仍被提前结束。`StagnationMonitor` 现已按照最大前进距离自适应放宽阈值，并在日志中标注停滞原因。
- **热点集中在 1-1 楼梯与 1-2 坑口**：JSONL 日志显示 `mario_x` 主要停留在 576±36 与 288±20 两个桶。重构后的 `MacroSequenceLibrary` 保留热点信息，使探索包装器能针对性插入前进组合。
- **时间上限未触发停滞导致刷分循环**：最新 `runs/episode_log.jsonl` 多次出现 `mario_x ≤ 40` 的回合通过反复踩敌人将 `stagnation_reason` 维持在 `no_progress` 并拖到 `time_limit`。新增 `idle_counter` 与记分缓冲上限，超过预算即判定为 `score_loop` 并截断，同时在日志写入 `stagnation_idle_frames` 便于复盘。
- **训练配置难以复现**：CLI 参数分散且无持久化。`TrainingConfig` 统一封装全部参数并导出 `run_config.json`，同时增加 `--diagnostics-log-interval` 方便调节诊断频率。
- **诊断信号分散**：`DiagnosticsLoggingCallback` 汇总平均前进距离、内在奖励与停滞原因比例；分析工具新增停滞原因统计与 JSON 导出，便于自动化处理。
- **包装器维护困难**：原 `wrappers.py` 集成绩效、宏动作与图像变换，维护成本高。现将动作、探索、观测处理拆分为独立模块，同时保留聚合入口兼容旧代码。

## 后续计划
- 利用 `stagnation_reason` 与新增 `stagnation_idle_frames` 统计构建自适应探索策略，在线调节 ε 与宏动作权重。
- 基于 `run_config.json` 提供批量实验与网格搜索工具，简化调参流程。
- 在分析工具中加入多实验对比与可视化脚本，自动展示热点演化趋势。
- 针对 `score_loop` / `time_limit` 比例联动探索包装器，自动加大热点区域前进宏序列权重。
- 完善文档与教程，示例 TensorBoard 诊断指标与 JSONL 二次分析方法。

## 推荐训练配置（RTX 2080 Ti + 12 vCPU + 40GB RAM）
针对 12 线程 CPU、11GB 显存的 RTX 2080 Ti 与 40GB 内存，我们建议以下参数组合：

```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
  --algo ppo --policy-preset mario_large \
  --total-timesteps 6000000 --num-envs 16 --vec-env subproc \
  --frame-skip 4 --frame-stack 4 --resize 84 84 \
  --reward-profile smb_progress --observation-type gray \
  --stagnation-frames 720 --stagnation-progress 1 \
  --exploration-epsilon 0.10 --exploration-final-epsilon 0.02 --exploration-decay-steps 2500000 \
  --entropy-coef 0.02 --entropy-final-coef 0.005 --entropy-decay-steps 2500000 \
  --icm --icm-eta 0.015 --icm-lr 5e-5 \
  --checkpoint-freq 150000 --diagnostics-log-interval 2000 \
  --episode-log episode_log.jsonl --tensorboard
```

推荐理由：
- `num_envs=16` 能充分利用 12 vCPU 并给 GPU 留出推理余量，`subproc` 方式降低单核阻塞。
- `stagnation-frames=720` 配合新的 `idle_counter`，既允许宏动作试探，又能更快识别刷分循环并触发 `score_loop`。
- `exploration` 与 `entropy` 采用 250 万步线性衰减，前期保持足够探索，后期趋向收敛策略。
- `icm` 配置以 5e-5 的学习率和 0.015 的内在奖励系数适配 2080 Ti 显存，兼顾探索与稳定训练。
- `diagnostics-log-interval=2000` 与 `episode_log` 结合，便于在 TensorBoard 与 JSONL 中同步观察新增加的停滞指标。

欢迎通过 Issue / PR 反馈需求，共同完善 FC Emulator Toolkit。
