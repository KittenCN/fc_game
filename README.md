# FC Emulator Toolkit

FC Emulator Toolkit 基于 [`nes-py`](https://github.com/Kautenja/nes-py) 构建，提供从 NES ROM 加载、Gymnasium 环境封装、策略库、训练脚本到日志分析的一体化工作流。近期完成的重构聚焦于环境抽象、探索机制、训练诊断与日志分析，使得在经典红白机游戏上的强化学习实验更稳定、更易复现。

## 项目亮点
- **模块化环境封装**：`fc_emulator.rl_env` 使用全新的 `AutoStartController` 与 `StagnationMonitor`，自动跳过开场动画并按照实际前进距离动态放宽停滞阈值，同时在 `info.metrics` 中写入 `stagnation_reason` 与热点信息。
- **动作/探索组件解耦**：动作模板、图像处理与宏动作探索分别迁移至 `actions.py`、`observation.py`、`exploration.py`，`wrappers.py` 仅作为兼容入口，便于扩展自定义策略组合。
- **结构化训练配置**：`train.py` 通过 `TrainingConfig` 统一封装环境、算法、探索与日志参数，启动时自动导出 `runs/run_config.json`，新增 `--diagnostics-log-interval` 控制诊断频率。
- **在线诊断回调**：`DiagnosticsLoggingCallback` 周期性输出平均前进距离、内在奖励与停滞原因比例至 TensorBoard（`diagnostics/*`），与 Episode JSONL 日志互补。
- **增强日志分析**：`fc_emulator.analysis` 现支持 `--json` 输出、停滞原因分布、最近窗口前进均值等统计，处理大体量 JSONL 时无需将全部数据载入内存。

## 目录结构
```
fc_emulator/
├── actions.py              # 离散动作模板与包装器
├── auto_start.py           # 标题画面自动跳过控制器
├── exploration.py          # 宏动作/热点驱动探索包装
├── observation.py          # 图像缩放与帧堆叠包装
├── stagnation.py           # 停滞检测与动态阈值模块
├── rl_env.py               # NESGymEnv 主环境
├── rl_utils.py             # 向量化环境构建与算法映射
├── callbacks.py            # 训练期回调（日志、诊断、熵/ε 调度）
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
- `--resize H W`：送入策略前的降采样尺寸，如 `--resize 84 84`
- `--action-set`：选择预设动作（`default`、`simple`、`smb_forward`）或自定义组合 `"RIGHT;A,RIGHT;B"`
- `--exploration-epsilon` / `--exploration-final-epsilon` / `--exploration-decay-steps`
- `--entropy-*`：配合 `EntropyCoefficientCallback` 线性衰减策略熵系数
- `--reward-profile`：`none` / `smb_progress` / `smb_dense`
- `--icm`：启用内在奖励模块（需像素观测）
- `--diagnostics-log-interval`：诊断信息写入间隔（默认 5000 步）

训练流程会自动：
1. 在 `runs/run_config.json` 存储完整配置以便复现；
2. 如存在断点，加载最新 `ppo_agent_*.zip`；
3. 将诊断指标写入 TensorBoard，并持续以 JSONL 记录每个 episode 的奖励/停滞信息。

### 推理示例
```bash
python -m fc_emulator.infer --rom roms/SuperMarioBros.nes \
  --model runs/ppo_agent_XXXXXX.zip --observation-type gray \
  --frame-stack 4 --episodes 3 --deterministic
```

## 日志与分析
- 训练时 `EpisodeLogCallback` 默认写入 `runs/episode_log.jsonl`，记录基础奖励、塑形奖励、内在奖励与 `metrics.stagnation_reason`。
- `DiagnosticsLoggingCallback` 会把平均 `mario_x`、内在奖励、停滞原因占比写入 TensorBoard（`diagnostics/*`）。
- 使用分析工具快速汇总：
  ```bash
  python -m fc_emulator.analysis runs/episode_log.jsonl --bucket-size 32 --top 10
  python -m fc_emulator.analysis runs/episode_log.jsonl --json  # 以 JSON 输出
  ```
  统计结果包含平均/中位/最大前进距离、最近 200 回合均值、停滞原因分布、热点区间、负回合比例、内在奖励与终止原因等。

## 近期问题与解决方案
- **训练频繁被停滞截断**：旧版停滞阈值固定 752 帧，导致中后期探索仍被提前结束。`StagnationMonitor` 现根据最大前进距离自适应放宽阈值，并在日志中标注停滞原因。
- **热点集中在 1-1 楼梯与 1-2 坑口**：JSONL 日志显示 `mario_x` 主要聚集在 576–736 与 288–320 桶。重构宏动作库（`MacroSequenceLibrary`）并保留热点信息，使探索包装器能针对性插入前进组合。
- **训练配置难以复现**：CLI 参数分散且无持久化。`TrainingConfig` 统一封装全部参数、导出 `run_config.json`，同时新增 `--diagnostics-log-interval` 调整诊断频率。
- **诊断信号分散**：新增 `DiagnosticsLoggingCallback` 汇总平均前进距离、内在奖励与停滞原因比例；分析工具新增停滞原因统计与 JSON 导出，方便自动化处理。
- **包装器维护困难**：原 `wrappers.py` 集成绩效、宏动作与图像变换，维护成本高。现将动作、探索、观测处理拆分为独立模块，同时保留聚合入口兼容旧代码。

## 后续计划
- 将 `stagnation_reason` 统计用于在线调节 ε/宏动作权重，构建自适应探索策略。
- 基于 `run_config.json` 提供批量实验与网格搜索工具，简化调参流程。
- 在分析工具中加入多实验对比与可视化脚本，自动展示热点演化趋势。
- 完善文档与教程，示例化 TensorBoard 诊断指标与 JSONL 二次分析方法。

欢迎通过 Issue / PR 反馈需求，共同完善 FC Emulator Toolkit。
