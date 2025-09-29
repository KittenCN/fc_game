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

### 2025-09-28 深夜（20 万步回归评估）

#### 发现的问题
- `episode_log_fast.jsonl` 共 1194 回合，平均长度 204 步，`mario_x` 均值仅 72，最大 860；≥512 的样本占比 4.4%。
- `stagnation_reason` 以 `backtrack` 为主（900 次，75%）；热点统计中 `bucket=0` 占 72%，说明代理仍大量回落至出生点。
- `stagnation_event` 中有 776 次被记为 `level_transition`，实际由强制 `backtrack_stop` 提前终止引发，导致 epsilon 提升逻辑未触发。
- 大量回合在达到最佳进度后立即被截断（`stagnation_frames≈4`，`stagnation_idle_frames≈580`），主因是回退阈值在较小进度也会触发，复位过于激进。

#### 分析与原因
- `StagnationMonitor` 未限制最小进度，导致刚跨出起点即触发 `backtrack_stop`，大量 episode 在出生点周围循环。
- `ExplorationEpsilonCallback` 仅根据 `stagnation_event` 判断热点重复，未覆盖被标记为 `level_transition` 的 backtrack-stop 回合，导致 ε 提升从未发生。

#### 采取的方案
- 引入最小进度门槛 `backtrack_stop_min_progress`（默认 128），仅当达到该进度后才允许强制回退截断。
- 强制截断时统一写入 `stagnation_event='backtrack_stop'`，避免与真实关卡切换混淆。
- `ExplorationEpsilonCallback` 同时监控 `stagnation_reason` 与 `stagnation_event`，并将触发种类扩展至 `backtrack_stop`/`score_loop`，阈值降至 3 次以更快提升 ε。

#### 实施改动
- 代码文件：`stagnation.py`、`rl_env.py`、`rl_utils.py`、`train.py`、`callbacks.py`
- 要点：
  - 新增 CLI 参数 `--stagnation-backtrack-stop-min` 并贯通配置/环境；
  - 在 `StagnationMonitor` 中校验最小进度并统一事件标签；
  - 更新 epsilon 调度逻辑以捕获 backtrack-stop，缩短触发阈值。

#### 下一步计划
- 在新逻辑下继续 400k~500k timestep 训练以观察热点扩散与 ≥512 样本占比变化（参数见下）。
- 关注 `docs/UPDATE_LOG.md` 中 `stagnation_event='backtrack_stop'` 的占比，确认是否显著下降。

### 2025-09-28 深夜（40 万步延长实验）

#### 发现的问题
- 40 万步运行（`episode_log_eval2.jsonl`，1781 回合）平均 `mario_x` 仅 22.6、最高 1168，≥512 桶占比 0.9%，整体前进显著退化。
- `stagnation_event='backtrack_stop'` 触发 997 次（56%），热点统计显示 `bucket=0` 占 91.5%，说明代理频繁返回出生点并被强制截断。
- `stagnation_frames` 均值 349、`stagnation_idle_frames` ≈ 611，回合在短暂进度后迅速回落，ε 提升不足以带来突破。

#### 分析与原因
- `backtrack_stop_min_progress=128` 仍过低，刚突破初期即会遭到强制截断，导致长程样本难以积累。
- 强制回退在第一次出现时即生效，缺乏再尝试机会，与高频回落叠加造成大量短回合。

#### 采取的方案
- 移除强制回退截断逻辑，改为根据回退次数发出 `backtrack_warning`，仅向探索与奖励提供信号。
- 将回退进度阈值提升至 256，并要求同一热点连续两次回退才触发警告；`backtrack_penalty_scale` 提升至 1.5。
- 在奖励塑形中改为“正向进度 + 闲置惩罚 + 回退扣分”简化模型，避免巨额负值；引入前进保持奖励与回退连击惩罚。
- `EpsilonRandomActionWrapper` 在收到多次回退告警时自动注入前进宏动作序列，形成“救援”机制。
- 新增最优模型检查点回调：按窗口平均 `mario_x` 保存最佳模型，并在连续窗口无改进时自动回滚到最佳权重继续训练。
- 受影响文件：`stagnation.py`、`rl_env.py`、`rl_utils.py`、`train.py`、`rewards.py`、`callbacks.py`、`exploration.py`。

#### 下一步计划
- 继续进行 600k timestep 训练，验证 `backtrack_warning` 占比是否下降、≥512 桶样本是否回升（命令见下）。
- 重点监控：`bucket=0` 次数、`mario_x` 均值/分位数、`stagnation_reason` 分布以及 ε 提升触发频率。

##### 建议命令
```bash
python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
  --algo ppo --policy-preset baseline \
  --total-timesteps 1200000 --num-envs 6 --vec-env subproc \
  --frame-skip 4 --frame-stack 4 --resize 84 84 \
  --reward-profile smb_progress --observation-type gray \
  --n-steps 1024 --batch-size 256 \
  --icm --icm-eta 0.015 --icm-lr 5e-5 --icm-feature-dim 128 --icm-hidden-dim 128 \
  --stagnation-frames 760 --stagnation-progress 1 \
  --stagnation-bonus-scale 0.15 --stagnation-idle-multiplier 1.1 \
  --stagnation-backtrack-penalty 1.5 \
  --max-episode-steps 3200 \
  --exploration-epsilon 0.08 --exploration-final-epsilon 0.02 --exploration-decay-steps 2000000 \
  --entropy-coef 0.02 --entropy-final-coef 0.0045 --entropy-decay-steps 3000000 \
  --checkpoint-freq 200000 --diagnostics-log-interval 2000 \
  --best-checkpoint best_agent.zip --best-metric-key mario_x --best-metric-mode max --best-window 30 --best-patience 6 --best-min-improve 1.0 \
  --episode-log episode_log_eval3.jsonl
```

### 2025-09-29 凌晨（最优回滚 + 奖励调节）

#### 发现的问题
- 60 万步实验 `episode_log_eval3.jsonl` 中，`mario_x` 均值虽提升至约 206，但 `stagnation` 占 99%，`backtrack` 警告仍占 26%，shaped reward 均值约 -2182，说明前进能力有所增强但奖励信号仍高度偏负，回退频繁。

#### 采取的方案
- 引入 `BestModelCheckpointCallback`，以窗口平均 `mario_x` 追踪并保存最优模型，在连续 `patience` 个窗口无改进时自动回滚继续训练。
- 调整奖励函数：时间罚项降至 0.003，回退连击惩罚降至 0.2，并新增统一 `reward_scale=0.1`，保持正向奖励与惩罚平衡。
- 强化探索救援：救援触发后暂时提高 ε 并重置冷却，增加打破热点滞留的几率。
- 默认为 PPO 开启 `normalize_advantage`，最佳模型窗口/耐心改为 30/6。
- 受影响文件：`train.py`、`callbacks.py`、`exploration.py`、`rewards.py`、`README.md`。

#### 下一步计划
- 使用上述命令重新训练，重点监控：`backtrack_warning` 比例、`mario_x` 分位、shaped reward 分布以及最优回滚触发频次。
- 若回退仍高，可继续缩短救援冷却或引入阶段性停滞阈值/里程碑奖励。

### 2025-09-29 上午（内存与奖励再平衡）

#### 发现的问题
- `VecFrameStackPixelsDictWrapper` 每步复制整个堆叠帧缓冲，回合终止时还会额外全拷贝一次；多进程 rollout 下这个热点导致主机内存和 CUDA 显存呈阶梯式增长。
- 最优模型触发回滚时直接 `algo_cls.load`，旧模型仍占用 CUDA 缓存；多次触发后 GPU reserved memory 未释放。
- ICM 默认 256 维编码、1e-4 学习率与 baseline `n_steps=1024` 在 11GB 显存机器上压力依旧较高；reward shaping 负项仍明显大于正项，shaped reward 积极样本不足。

#### 分析与原因
- `stacked_obs.copy()` 与 `prev_stacked = self.stacked_obs.copy()` 在每个 env step 分配新的 ndarray，Python allocator 复用缓慢。
- 回滚逻辑未显式释放旧模型或清理 CUDA cache，导致空闲显存不回收。
- ICM 规模与 reward 权重与当前训练现状不匹配，主导了负值优势。

#### 采取的方案
- 代码变更：
  - `fc_emulator/observation.py`：改为复用堆叠缓冲，使用 `np.copyto` + 针对终止 env 的按需复制，消除逐帧分配。
  - `fc_emulator/train.py`：在最优回滚前后执行 `gc.collect()` 和 `torch.cuda.empty_cache()`，并刷新回调对象；CLI 新增 `--icm-device`，默认 ICM 采用轻量配置。
  - `fc_emulator/icm.py`：默认特征/隐层降至 128，学习率降至 5e-5，`device` 参数支持 `cuda:idx` 并做校验。
  - `fc_emulator/policies.py`：baseline preset 调整为 `n_steps=768`、`batch_size=192`，降低 rollout buffer 占用。
  - `fc_emulator/rewards.py`：提升前进正奖励与里程碑权重，下调时间/停滞/死亡惩罚，限制回退连击惩罚上限，`reward_scale` 提升至 0.12。
- 运行命令（20 万步回归 / 6 环境）：
  ```bash
  python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
    --algo ppo --policy-preset baseline --total-timesteps 200000 \
    --num-envs 6 --vec-env subproc --frame-skip 4 --frame-stack 4 --resize 84 84 \
    --reward-profile smb_progress --icm --icm-device auto \
    --checkpoint-freq 200000 --episode-log episode_log_light.jsonl --tensorboard
  ```

#### 实验与结果
- 待运行新实验采集：GPU reserved memory 曲线、RSS 变化、`mario_x` 均值/95 分位、`intrinsic_reward` 均值、`stagnation_reason` 分布。
- 预期：显存不再阶梯式上升，RSS 进入平稳波动区；shaped reward 分布较前一版更接近零点，停滞占比下降。

#### 后续计划
- 若内存仍缓慢攀升，考虑在像素堆叠阶段引入 uint8→float16 的懒转换或 Torch 缓冲池复用。
- 对新日志进行奖励分项拆解，按需暴露 YAML 权重配置。
- 基于热点统计实现存档回放或子目标探索，尝试 Go-Explore 式策略。

### 2025-09-29 下午（网络路线调研）

#### 发现的问题
- 原调研仅覆盖三类方案，缺乏对 Rainbow/RND/Go-Explore 等方向的证据比勘，短期路线也未明确优先级。

#### 采取的方案
- 更新 `docs/NETWORK_RESEARCH.md`：新增“一句话结论”，扩展方案 A–G 分析，修正 Rainbow replay 内存估算（≥7GiB）、强调 RecurrentPPO 序列对齐、UNREAL 权重调度、RND 共享编码、Go-Explore savestate 依赖等要点。
- 明确路线规划：短期落地 IMPALA-ResNet-LSTM + 共享编码 RND；中期在 Conv-Transformer 与 Rainbow-DQN++ 中择优；长期预留 Go-Explore、层次 RL、Dreamer/MuZero。

#### 后续计划
- 按调研“后续工作项”推进：优先实现 RecurrentPPO + RND 基线，并准备 Rainbow 序列回放与 Go-Explore savestate 技术验证。
