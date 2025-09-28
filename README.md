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
  --observation-type gray --diagnostics-log-interval 2000
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
- `fc_emulator.hotspot_monitor` 可持续轮询 JSONL 日志，实时输出热点分布、停滞与终止占比，适合训练中监控：
  ```bash
  python -m fc_emulator.hotspot_monitor --log runs/monitor_run_v2/episode_log.jsonl --bucket-size 32 --top 8 --oneshot
  python -m fc_emulator.hotspot_monitor --log runs/monitor_run_v2/episode_log.jsonl --poll-interval 10
  ```

## 推荐训练配置（RTX 2080 Ti + 12 vCPU + 40GB RAM）
针对 12 线程 CPU、11GB 显存 RTX 2080 Ti 与 40GB RAM，我们推荐如下组合：

```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
  --algo ppo --policy-preset baseline \
  --total-timesteps 1200000 --num-envs 8 --vec-env subproc \
  --frame-skip 4 --frame-stack 4 --resize 84 84 \
  --reward-profile smb_progress --observation-type gray \
  --stagnation-frames 720 --stagnation-progress 1 \
  --stagnation-bonus-scale 0.15 --stagnation-idle-multiplier 1.1 \
  --stagnation-backtrack-penalty 1.5 \
  --exploration-epsilon 0.08 --exploration-final-epsilon 0.02 --exploration-decay-steps 3000000 \
  --entropy-coef 0.02 --entropy-final-coef 0.0045 --entropy-decay-steps 3000000 \
  --checkpoint-freq 200000 --diagnostics-log-interval 2000 \
  --best-checkpoint best_agent.zip --best-window 30 --best-patience 6 \
  --episode-log episode_log.jsonl
```

推荐理由：
- `num_envs=30` 精准匹配 vCPU 数，避免 32 并发造成的上下文切换开销，并确保每步 rollout 都能及时回传热点统计。
- `resize 84 84` 为更深的 `MarioFeatureExtractor` 提供充足信息量，同时在 11GB 显存下仍能容纳 12 并行环境与 ICM。
- `exploration` / `entropy` 衰减延长至 300 万步，利用热点持久化策略逐步降低随机性但保留宏动作注入窗口。
- `stagnation-frames=720` 搭配持久化热点与 `score_loop` 监测，让宏动作有尝试空间同时快速截断刷分循环。
- `checkpoint-freq=1000000` 与频率更高的诊断刷新（2000）确保能观察热点分布与 `stagnation_reason` 变化并及时回滚。

欢迎通过 Issue / PR 反馈需求，共同完善 FC Emulator Toolkit。

## 最佳模型回滚（可选）
- 通过 `--best-checkpoint best_agent.zip` 启用最优模型保存；默认以最近 20 个 episode 的 `mario_x` 平均值衡量表现，并在连续 5 个窗口无改进时回退到最佳权重继续训练。
- 可使用 `--best-metric-key`、`--best-window`、`--best-patience`、`--best-min-improve` 调整指标来源、窗口大小、耐心与最小改进幅度，满足不同实验需求。
- 最优模型文件存放在 `log_dir` 下指定路径，可与常规 checkpoint 配合使用，便于快速回溯。

## 更新维护记录
- 近期问题、实验结果与后续计划已迁移至 [`docs/UPDATE_LOG.md`](docs/UPDATE_LOG.md)，请在更新诊断信息或实验结论时同步维护该文件。
