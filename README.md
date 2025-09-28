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
- `fc_emulator.hotspot_monitor` 可持续轮询 JSONL 日志，实时输出热点分布、停滞与终止占比，适合训练中监控：
  ```bash
  python -m fc_emulator.hotspot_monitor --log runs/monitor_run_v2/episode_log.jsonl --bucket-size 32 --top 8 --oneshot
  python -m fc_emulator.hotspot_monitor --log runs/monitor_run_v2/episode_log.jsonl --poll-interval 10
  ```

## 近期问题与解决方案
- **停滞终止比例过高，热点分布极度集中**：`runs/episode_log.jsonl` 共 2503 回合，其中 2477 次因 `stagnation` 提前结束，热点集中在 `0`、`32`、`64` 桶（占比>45%），暴露出生点徘徊问题。
- **热点记忆随 reset 清零**：旧版 `EpsilonRandomActionWrapper` 在 `reset()` 时清空热点缓存。现已改为跨回合保留热点计数、方向映射并自动刷新 `_active_hotspot`，宏动作可以持续针对历史卡点发力。
- **热点强度缺乏自适应加权**：首次 12 并行实验（`runs/monitor_run`，146 回合）虽出现 `mario_x=725` 的长程样本，0~64 桶仍占 48.4%。新增 `hotspot_intensity` 权重后（`runs/monitor_run_v2`，145 回合），`mario_x` 均值提升至 153.6，最大值达 816，热点开始向 `224~320` 桶扩散，但停滞率仍为 100%。
- **CPU 线程过载**：`runs/run_config.json` 使用 `num_envs=32`，在 12 vCPU 机器上造成频繁上下文切换。推荐配置降至 12 并配合更长的探索衰减以稳定样本效率。
- **刷分循环仍偶发**：`stagnation_reason='score_loop'` 虽仅 1 次，但 `stagnation_idle_frames` 长期接近阈值，后续需结合热点比率抑制刷分循环。
- **诊断与监控工具链完善**：`fc_emulator.hotspot_monitor` 支持实时轮询 JSONL，搭配 TensorBoard 诊断，可快速验证重构效果与热点迁移路径。

## 后续计划
- 利用 `hotspot_intensity` 指标进一步联动 ε/熵调度与内在奖励，降低 0~64 桶滞留占比。
- 在 `analysis` 中加入多 run 对比与可视化脚本（聚合 `runs/*/episode_log.jsonl`），观察热点迁移趋势。
- 针对高 idle 的热点自动加大宏序列注入与内在奖励，持续抑制刷分循环。
- 基于 `run_config.json` 提供批量实验 / 网格搜索脚本，支撑参数扫描。
- 完善文档与教程，示例 TensorBoard 诊断与 `hotspot_monitor` 监控方法。

## 推荐训练配置（RTX 2080 Ti + 12 vCPU + 40GB RAM）
针对 12 线程 CPU、11GB 显存 RTX 2080 Ti 与 40GB RAM，我们推荐如下组合：

```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
  --algo ppo --policy-preset mario_large \
  --total-timesteps 5000000 --num-envs 16 --vec-env subproc \
  --frame-skip 4 --frame-stack 4 --resize 84 84 \
  --reward-profile smb_progress --observation-type gray \
  --stagnation-frames 840 --stagnation-progress 1 \
  --exploration-epsilon 0.08 --exploration-final-epsilon 0.02 --exploration-decay-steps 2000000 \
  --entropy-coef 0.02 --entropy-final-coef 0.0045 --entropy-decay-steps 3000000 \
  --icm --icm-eta 0.015 --icm-lr 5e-5 \
  --checkpoint-freq 1000000 --diagnostics-log-interval 2000 \
  --episode-log episode_log.jsonl --tensorboard
```

推荐理由：
- `num_envs=16` 精准匹配 vCPU 数，避免 32 并发造成的上下文切换开销，并确保每步 rollout 都能及时回传热点统计。
- `resize 84 84` 为更深的 `MarioFeatureExtractor` 提供充足信息量，同时在 11GB 显存下仍能容纳 16 并行环境与 ICM。
- `exploration` / `entropy` 衰减延长至 300 万步，利用热点持久化策略逐步降低随机性但保留宏动作注入窗口。
- `stagnation-frames=720` 搭配持久化热点与 `score_loop` 监测，让宏动作有尝试空间同时快速截断刷分循环。
- `checkpoint-freq=1000000` 与频率更高的诊断刷新（2000）确保能观察热点分布与 `stagnation_reason` 变化并及时回滚。

欢迎通过 Issue / PR 反馈需求，共同完善 FC Emulator Toolkit。

## 更新记录（2025-09-28）

### 发现的问题
- `runs/episode_log.jsonl` 共 3154 回合，其中 3126 次因停滞提前结束，`stagnation_event='backtrack'` 占比 61.7%，热点集中在 `0~96` 桶。
- 停滞回合后热点方向被标记为 `backward`，导致宏动作探索持续注入后退序列，代理在出生点反复徘徊。

### 分析与原因
- `EpsilonRandomActionWrapper` 依据最近位移方向更新 `_hotspot_direction_map`；停滞触发时 `mario_x` 回落，使 `_recent_direction` 变为 `backward`。
- 热点方向错误地指向 `backward`，`_get_priority_sequences` 优先采样后退宏序列，加剧了停滞与回退事件。

### 采取的方案
- 代码变更摘要：
  - 文件/模块：`fc_emulator/exploration.py`
  - 关键改动：新增 `_determine_hotspot_direction`，在停滞/回退事件下强制热点方向指向 `forward`，避免重复注入后退宏动作。
- 运行命令与参数：
  ```bash
  python -m compileall fc_emulator/exploration.py
  # 语法检查已通过；后续需重新运行训练脚本评估效果
  ```

### 实验与结果
- 数据集/切分：暂未重新训练，等待新一轮 PPO 运行验证热点分布与 `mario_x` 长度尾部。
- 指标（基线 vs 新方案）：预计需对比停滞终止占比、`stagnation_event` 方向分布与 `mario_x` 95% 分位数。
- 资源占用与耗时：待后续实验记录。
- 结论：已消除热点方向反馈的显性缺陷，需通过短程回归实验确认探索质量改善幅度。

### 后续计划
- 进行 ≥200k timestep 的短程 PPO 训练，观察热点方向分布与 `backtrack` 事件占比（负责人：开放）。
- 若回退仍占主导，考虑在 `MacroSequenceLibrary` 中拆分中立序列或调整停滞阈值以匹配新策略（优先级：中）。
- 更新 `fc_emulator.analysis`，增加按时间片展示热点方向的能力（优先级：低）。
