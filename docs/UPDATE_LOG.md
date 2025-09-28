# FC Emulator 更新维护记录

本文件汇总近期问题诊断、解决方案、实验结论与后续计划，便于持续迭代与复盘。

## 近期问题与解决方案
- **停滞终止比例过高，热点分布极度集中**：`runs/episode_log.jsonl` 共 2503 回合，其中 2477 次因 `stagnation` 提前结束，热点集中在 `0`、`32`、`64` 桶（占比>45%）。
- **热点记忆随 reset 清零**：旧版 `EpsilonRandomActionWrapper` 在 `reset()` 时清空热点缓存，宏动作无法跨回合持续发力。
- **热点强度缺乏自适应加权**：12 并行实验（`runs/monitor_run_v2`）虽出现长程样本，0~64 桶仍占 48.4%，需要针对历史卡点加权。
- **CPU 线程过载**：`num_envs=32` 在 12 vCPU 机器上触发频繁上下文切换，样本效率下降。
- **刷分循环仍偶发**：`stagnation_reason='score_loop'` 虽占比低，但 `stagnation_idle_frames` 长期接近阈值。
- **诊断与监控工具链完善**：`fc_emulator.hotspot_monitor` 加入实时轮询后，可结合 TensorBoard 快速定位热点迁移。

## 后续计划（滚动）
- 利用 `hotspot_intensity` 联动 ε / 熵调度与内在奖励，降低 0~64 桶滞留占比。
- 在 `analysis` 中加入多 run 对比脚本，跟踪热点迁移趋势。
- 面向高 idle 热点加强宏序列注入与内在奖励，抑制刷分循环。
- 基于 `run_config.json` 搭建批量实验脚本支撑参数扫描。
- 持续完善文档与教程，沉淀 TensorBoard 与 `hotspot_monitor` 使用示例。

## 更新记录

### 2025-09-28

#### 发现的问题
- `runs/episode_log.jsonl` 共 3154 回合，其中 3126 次因停滞提前结束，`stagnation_event='backtrack'` 占比 61.7%，热点集中在 `0~96` 桶。
- 停滞回合后热点方向被错误标记为 `backward`，导致宏动作探索持续注入后退序列。

#### 分析与原因
- `EpsilonRandomActionWrapper` 依据最近位移方向更新 `_hotspot_direction_map`，停滞触发时 `mario_x` 回落导致 `_recent_direction` 变为 `backward`。
- 热点方向指向 `backward` 使 `_get_priority_sequences` 优先采样后退宏序列，加剧停滞问题。

#### 采取的方案
- 关键改动：
  - 文件/模块：`fc_emulator/exploration.py`
  - 新增 `_determine_hotspot_direction`，停滞/回退事件下强制热点方向指向 `forward`。
- 运行命令：
  ```bash
  python -m compileall fc_emulator/exploration.py
  ```

#### 实验与结果
- 数据集/切分：暂未重新训练，等待新一轮 PPO 验证热点分布与 `mario_x` 尾部。
- 指标：关注停滞终止占比、热点方向分布与 `mario_x` 95% 分位数。
- 资源占用：待补充。
- 结论：热点方向反馈缺陷已消除，需短程回归确认探索改善。

#### 后续计划
- 启动 ≥200k timestep 短程训练，观察热点方向与 `backtrack` 占比。
- 若回退仍主导，考虑拆分中立序列或调节停滞阈值。
- 在 `analysis` 中增加时间片热点分布可视化。

### 2025-09-28 晚间

#### 发现的问题
- 新一轮 5M timestep 训练中 `time_limit` 终止 669 次（占 26%），34.6% 回合长度 ≥2000，平均 `mario_x≈361`。
- 后半程 `backtrack` 事件 541 次，超过 `stagnation` 522 次，热点迁移至 `384/288/576` 桶但 0~96 桶仍复发。
- 平均 shaped reward -105，`stagnation_frames` 均值 974，长回合空跑拉低优势。

#### 分析与原因
- `bonus_scale=0.25` 将高进度阈值抬升至 1000+ 帧，回退时未及时截断，episode 被 `TimeLimit` 终止。
- ε 衰减至 0.02 后，热点重复命中未触发宏动作提升，`backtrack` 难以摆脱。
- 时间罚项与停滞罚项叠加，长程样本 Reward 仍为负，策略倾向早结束。

#### 采取的方案
- 文件/模块：`stagnation.py`、`rl_env.py`、`rl_utils.py`、`callbacks.py`、`rewards.py`、`train.py`
- 关键改动：
  - 暴露并调优停滞参数（`bonus_scale`、`idle_multiplier`、`backtrack_penalty`、`backtrack_stop_ratio`），默认停滞帧数降至 760、最大 episode 步数降至 3200。
  - 对热点重复停滞/回退回合临时提升 ε，并扩充前进宏序列，强化热点突破。
  - 奖励重平衡：降低时间罚项，引入里程碑奖励，并对高桶位停滞罚项做衰减。
- 运行命令：
  ```bash
  python -m compileall fc_emulator
  ```

#### 实验与结果
- 数据集/切分：待以 200k timestep 快速实验验证长回合比例与热点扩散。
- 指标：重点监控 `mario_x` 均值、`backtrack` 占比、`stagnation_frames` 均值。
- 资源占用：实验启动后补充。
- 结论：已完成停滞与探索策略的防御性调整，需回归实验确认收益。

#### 后续计划
- 启动 20 万 timestep 快速实验，评估 `time_limit` 占比与热点强度变化。
- 追加热点重复命中统计及可视化。
- 若奖励仍偏负，引入正归一化或 advantage clipping。

##### 快速实验命令
```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
  --algo ppo --policy-preset mario_large \
  --total-timesteps 200000 --num-envs 12 --vec-env subproc \
  --frame-skip 4 --frame-stack 4 --resize 84 84 \
  --reward-profile smb_progress --observation-type gray \
  --stagnation-frames 760 --stagnation-progress 1 \
  --stagnation-bonus-scale 0.15 --stagnation-idle-multiplier 1.1 \
  --stagnation-backtrack-penalty 1.0 --stagnation-backtrack-stop-ratio 0.7 \
  --max-episode-steps 3200 \
  --exploration-epsilon 0.08 --exploration-final-epsilon 0.02 --exploration-decay-steps 2000000 \
  --entropy-coef 0.02 --entropy-final-coef 0.0045 --entropy-decay-steps 3000000 \
  --icm --icm-eta 0.015 --icm-lr 5e-5 \
  --checkpoint-freq 200000 --diagnostics-log-interval 2000 \
  --episode-log episode_log_fast.jsonl --tensorboard
```
