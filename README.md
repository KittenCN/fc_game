# FC Emulator Toolkit

FC Emulator Toolkit 提供一套基于 Python 的 FC（NES）模拟器工具，可加载 `.nes` ROM 进行试玩，也支持构建强化学习（RL）训练流水线。项目基于 [`nes-py`](https://github.com/Kautenja/nes-py) 扩展，补充了控制器映射、渲染、环境封装、宏动作等组件，便于快速迭代智能体。

## 核心特性
- 支持加载 iNES 格式 ROM，提供键盘映射与实时渲染。
- 暴露屏幕像素、RAM 等观测信息，适配多种 RL 需求。
- 提供 Gymnasium 兼容接口，可直接接入 Stable-Baselines3、sb3-contrib 等算法实现。
- 内置训练 / 推理 CLI、离散动作封装、奖励函数预设、日志分析脚本。

## 环境准备
```bash
pip install -e .
# 如需强化学习组件
pip install -e .[rl]
```
若只缺图像缩放依赖，可单独安装 `pip install pillow`。

## 快速体验
```bash
python -m fc_emulator.cli --rom path/to/game.nes
```
默认键位：方向键 `WASD`，`A`→`J`，`B`→`K`，`Start`→`Enter`，`Select`→`Right Shift`。

## 强化学习训练工作流
```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
    --algo ppo --total-timesteps 1000000 --tensorboard \
    --num-envs 6 --vec-env subproc --frame-skip 4 --frame-stack 4
```
常用参数提示：
- **观测设置**：`--observation-type` 支持 `rgb`/`gray`/`ram`，新增 `rgb_ram`/`gray_ram`（像素+RAM）；`--resize H W` 可降低输入分辨率（如 `--resize 84 84`）。
- **动作集合**：`--action-set` 选择 `default`、`simple`，或传入自定义组合（例如 `"RIGHT;A,RIGHT;B"`）。
- **探索与停滞**：通过 `--exploration-epsilon` 及衰减参数调节 ε-greedy；`--stagnation-frames` / `--stagnation-progress` 控制停滞触发阈值。
- **宏动作 / 奖励**：可在 `fc_emulator/rewards.py` 中选择 `--reward-profile smb_dense` / `smb_progress` 等预设。

### 推理
```bash
python -m fc_emulator.infer --rom roms/SuperMarioBros.nes \
    --model runs/ppo_agent_xxx.zip --observation-type rgb_ram --frame-stack 4
```
推理默认启用 `render_mode="human"`，若训练时做过下采样，请保持相同的 `--resize` 设置。

## 目录结构一览
- `fc_emulator/`
  - `rl_env.py`：Gymnasium 环境封装与停滞逻辑。
  - `wrappers.py`：离散动作、观测缩放、宏动作与向量环境扩展。
  - `rl_utils.py`：统一构建向量环境、解析动作集合、算法映射。
  - `policies.py`：CNN / 多模态 / LSTM 策略预设。
  - `rewards.py`：SMB 奖励塑形配置。
  - `train.py` / `infer.py`：训练与推理 CLI。
  - 其它模块提供模拟器核心（`emulator.py`、`controller.py`、`renderer.py` 等）与日志分析工具。
- `examples/`：`human_play.py`、`random_agent.py` 等示例脚本。

## 最新更新（2025-09-27）
- **多模态观测**：环境新增 `rgb_ram` / `gray_ram` 类型，返回包含 `pixels` 与 `ram` 的字典观测；向量环境自动完成转置、帧堆叠。
- **策略预设**：新增 `mario_dual*` 系列（MultiInputPolicy）与 `mario_*_lstm` 预设，支持结合 RAM 与 LSTM 特征；配合 `--algo rppo`（sb3-contrib RecurrentPPO）即可启用循环策略。
- **内在奖励**：实现 `ICMVecEnvWrapper`，训练时传入 `--icm` 及 `--icm-*` 参数即可叠加好奇心奖励，日志中会记录 `diagnostics.intrinsic_reward`。
- **环境改进**：停滞检测会识别分数、关卡切换、能力提升等事件，减少误判；Episode 日志聚合整局奖励，便于分析。

## 训练观察与建议
- 过往日志显示智能体在 1-1 第一根水管附近容易卡死，900 帧停滞占比高。建议启用多模态观测并结合宏动作。
- 使用 ICM 时需关注外在奖励是否被抑制，可适当调整 `--icm-eta` / `--icm-beta`，或阶段性启用内在奖励。
- `python -m fc_emulator.analysis runs/episode_log.jsonl` 可输出均值 / 中位数 / 停滞帧数等统计指标，帮助判断训练是否陷入局部最优。

## 下一步计划 / TODO
- 对 ICM 与宏动作做系统消融实验，评估探索奖励对最终成绩的影响。
- 引入可视化脚本，自动回放最近若干停滞区间，定位策略缺陷。
- 规划 save-state、网络对战等模拟器层面的扩展。
- 深入探索 LSTM / Transformer 等长序列建模方案，提升通关稳定性。
