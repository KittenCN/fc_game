# FC Emulator 网络架构重构调研（2025-09-29）

## 0. 一句话结论
在当前单机资源与“尽量沿用 Stable-Baselines3”约束下，首要路线是采用 **IMPALA 风格残差卷积 + 轻量 LSTM 的 RecurrentPPO**，并将 **RND 内在奖励与策略共享同一个编码器（在 RND 头前阻断梯度）**；若该组合仍无法突破探索瓶颈，再考虑引入 **Go-Explore 风格的确定性探索阶段（可借助 Gym Retro savestate）** 作为下一阶升级。

本调研聚焦于当前 `CnnPolicy`/`MarioFeatureExtractor` 的局限，整理多条可行的重构路线，分析其收益、代价与实施步骤，供后续规划参考。

## 1. 基线回顾
- **视觉编码**：现有网络为 3–4 层卷积 + ReLU + BatchNorm，缺乏残差/下采样策略，远场信息易被压缩丢失。
- **时序感知**：仅依赖帧堆叠，无显式记忆；在长时间停滞或回退时无法利用早期上下文。
- **多模态融合**：`MarioDualFeatureExtractor` 在最终 MLP 才拼接 RAM 特征，像素/状态交互有限。
- **探索激励**：ICM 与策略分别编码像素，显存重复；仅提供 step-level 新颖度，缺少长期子目标。

## 2. 约束与目标
- 硬件：单卡 RTX 2080 Ti（11GB）、12 vCPU、40GB RAM；期望 `num_envs=6~8` 与 `n_steps≤1024`。
- 兼容性：优先保留 Stable-Baselines3 流程，可接受自定义 policy 但避免完全重写训练框架。
- 目标：提升探索能力、样本效率与稳定性，同时控制显存/内存开销。

## 3. 候选方案详细分析

### 方案 A：IMPALA/ResNet 卷积编码 + LSTM（单分支）
- **思路**：使用 IMPALA 风格残差块（3 级 stride-2 下采样）生成 256 维特征，再接 1–2 层 LSTM（隐藏维 256）。
- **优点**：残差提升梯度流动，LSTM 处理回退/停滞；可直接切换至 SB3 `RecurrentPPO`，复用现有回调。
- **风险**：
  - SB3 `RecurrentPPO` 要求 rollout 与 batch 序列对齐，并正确维护 LSTM 隐状态，否则训练会不稳定或吞吐显著下降（详见 sb3-contrib 文档）。
  - 需实现自定义 `features_extractor` 并协调 RND/ICM 共享编码，避免重复卷积。
- **资源**：约 5.5M 参数；FP16 显存 ~2.1GB（batch 128–160）。

### 方案 B：Conv-Transformer Actor-Critic（像素 + 自注意力）
- **思路**：将 84×84 灰度帧划分为 6×6 patch，映射到 128 维 token，堆叠 4 层轻量 Transformer；actor/critic 在 CLS token 上接 MLP。
- **优点**：自注意力覆盖长距离依赖，易与 RAM token 融合；结构具扩展性，可引入子目标 token。
- **风险**：注意力复杂度 O(N²)，需严格控制 token 数；SB3 无原生 Transformer policy，需要自定义 actor-critic 与掩码逻辑；训练对正则化敏感。
- **资源**：约 8.5M 参数；FP16 显存 ~2.5GB（batch 128），训练耗时较方案 A 增加 20–30%。

### 方案 C：Rainbow-DQN++（Mario 定制强化版）
- **思路**：在 DQN 基础上同时引入 Double、Dueling、优先级回放（PER）、NoisyNet、n-step 回报、分布式回报分布（C51/QR-DQN），并叠加 RND 内在奖励；尾部加轻量 LSTM（R2D2 风格）。
- **优点**：样本效率高（回放+分布式 Bellman），Noisy/RND 强化探索；动作离散，继承 Atari 经验；单机即可运行。
- **风险**：模块多、调参重；需实现 PER + LSTM 序列采样与 Noisy reset；极稀疏奖励仍需课程/起点复位；模型/缓冲保存复杂。
- **资源**：约 10M 参数；FP16 显存 ~2.5GB（batch 64）；按 84×84 灰度帧（uint8）计算，100 万帧约 6.6GiB，再加动作/奖励/索引与 PER 结构，整体需 ≥7GiB RAM。
- **实现细节提醒**：使用 LSTM 时需序列采样 + burn-in（R2D2 风格）更新隐藏态，并处理参数滞后；若扩展 Ape-X/R2D2，则需多 actor 优先级上报与隐藏态同步。

### 方案 D：PPO / IMPALA + RND + UNREAL 辅助任务
- **思路**：以 PPO（实现简单）或 IMPALA（高吞吐）为主干，叠加 RND/ICM 内在奖励与 UNREAL 辅助任务（像素控制、奖励预测、重建），并配 LSTM。
- **优点**：策略梯度更新稳定，易扩展到多进程；UNREAL 提高表征质量；RND 补足探索。
- **风险**：辅助损失/内在奖励需精调，建议以小权重暖启动并监控主任务奖励；IMPALA 需搭建 actor/learner 管线并实现 V-trace；样本效率仍逊于 replay 型方法。
- **资源**：~6M 参数；FP16 显存 ~1.8GB；CPU 需 8–12 worker。

### 方案 E：Go-Explore 两阶段探索
- **思路**：阶段一在确定环境中基于 cell 覆盖挖掘新状态并保存 savestate，阶段二（Robustification）在随机环境中通过行为克隆或 RL 训练鲁棒策略。
- **优点**：硬探索能力极强，可保留进展；与任意 RL 主干兼容。
- **风险**：需 emulator savestate 支撑（nes-py 需扩展或迁移 Gym Retro）；数据与状态管理复杂；鲁棒化阶段仍需大量训练。
- **资源**：探索阶段 CPU 为主；savestate 存储数百上千份约 2–5GB。

### 方案 F：层次/目标条件 RL + 势函数奖励
- **思路**：构建 Manager-Worker 两层结构（FeUdal、HIRO 等），Manager 制定目标（如 `x_pos` 提升 64、进入管道），Worker 以 UVFA/GC policy 执行；奖励采用势函数 shaping 保持最优策略不变。
- **优点**：长程 credit assignment 更轻松，可复用环境 info（`mario_x`、`flag_get`）；与 PPO/Rainbow 等主干组合。
- **风险**：目标空间定义与切换策略复杂；层次策略训练易不稳定，需要离策略校正。
- **资源**：参数量与主干接近（<8M），显存增量小；需额外目标编码。

### 方案 G：世界模型 + 行为策略（Dreamer-lite / MuZero-lite）
- **思路**：学习潜在动力学模型（Dreamer 系列）或隐式模型 + 规划（MuZero），通过 imagination 或 MCTS 提前规划。
- **优点**：样本效率潜力大，可学习抽象时空表征，适合稀疏奖励。
- **风险**：实现复杂，需要重构训练流水线、重放缓冲与稳定技巧；调试周期长，对小团队投入高。
- **资源**：~12M 参数；FP16 显存 ~3GB；额外 RAM（重放缓冲 1–2GB）。

## 4. 综合对比

| 方案 | 参数量 (M) | 显存需求* | 额外内存 | 实施复杂度 | 预期收益 | 主要风险 |
|------|-----------|-----------|----------|------------|----------|----------|
| A IMPALA-ResNet + LSTM | ~5.5 | ~2.1GB | 低 | ★★☆ | 样本效率↑、停滞鲁棒 | Recurrent 训练调参、共享编码改造 |
| B Conv-Transformer | ~8.5 | ~2.5GB | 低 | ★★★ | 长距离感知、多模态 | Attention 稳定性、实现工作量 |
| C Rainbow-DQN++ | ~10 | ~2.5GB | ≥7GB | ★★★☆ | 样本效率高、探索增强 | 模块复杂、缓冲管理、稀疏奖励仍难 |
| D PPO/IMPALA+RND+UNREAL | ~6 | ~1.8GB | 中 | ★★☆ | 更新稳定、吞吐高 | 辅助损失/内在奖励协同难 |
| E Go-Explore | 视主干 | 取决于鲁棒阶段 | 2–5GB | ★★★★ | 硬探索极强 | 依赖 savestate、工程量大 |
| F 层次/目标条件 RL | <8 | 与主干相同 | 低 | ★★★ | 长程规划、势函数保证 | 目标设计难、训练不稳 |
| G Dreamer-lite/MuZero-lite | ~12 | ~3.0GB | 1–2GB | ★★★★ | 模型式规划、稀疏奖励友好 | 流水线大改、调试成本高 |

*显存以 FP16、batch 128（DQN 为 64）估算，具体值视实现与并行度而定。

## 5. 方案择优建议
- **首选路线（短期 1–2 周）**：实现方案 A（IMPALA-ResNet-LSTM），并将 RND 连接到共享编码器取代 ICM，建立稳定且具记忆能力的 RecurrentPPO 基线；UNREAL 先暂缓，待主干收敛后再评估是否引入。
- **强化探索（中期 3–5 周）**：根据短期结果在方案 C（Rainbow-DQN++）与方案 B（Conv-Transformer）间择一推进：
  - 若需要更高样本效率与离线重播，优先 Rainbow-DQN++。
  - 若需要更强感知与多模态融合，选择 Conv-Transformer。
- **硬探索与长期规划（长期 >1 月）**：在前述方案仍难突破关卡时，引入方案 E（Go-Explore）或方案 F（层次 RL）；若有团队资源，可探索方案 G（Dreamer/MuZero）原型。

## 6. 关键实现提示
- **共享编码**：无论方案 A/B/D，建议优先让 RND 复用策略编码器（在 RND 预测头前 `stop_gradient`），必要时再加入 ICM，避免重复卷积并控制显存。
- **混合精度与显存**：所有方案优先支持 AMP，配合梯度累积/检查点以控制峰值显存。
- **日志与诊断**：扩展现有 `DiagnosticsLoggingCallback`，记录不同方案特有指标（例如 Rainbow 的 TD 分布 KL、Go-Explore 的 cell 覆盖度）。
- **回滚策略**：保留现有最优模型回滚逻辑，针对 replay 类方案需同步保存缓冲摘要（可采样子集）。

## 7. 实验设计草案
- **指标**：`mario_x` 均值/95 分位、停滞占比、显存峰值、wall-clock、intrinsic reward 均值、探索热点覆盖度。
- **流程**：
  1. 以现有 CnnPolicy 为基线运行 200k steps。
  2. 方案 A 与基线对照，记录优势。
  3. 根据结果选择方案 C 或 B 继续对照实验。
  4. 若仍卡关，评估方案 E/F 小规模原型。
- **判定标准**：在相同资源下，若方案平均 `mario_x` 提升 ≥20%、停滞占比下降 ≥15%、显存峰值 <9GB，则纳入主线。

## 8. 后续工作项
- [x] 设计 `ImpalaResidualFeatureExtractor` + `impala_lstm` 预设，并在训练流程中默认提供独立轻量 CNN 的 RND（可选 `--rnd-shared-encoder`）。
- [ ] 评估方案 D 的 RND/UNREAL 协同，明确损失权重与日志需求。
- [ ] 规划 Rainbow-DQN++ 的模块拆分（PER、Noisy、RND、LSTM 序列采样），确认复用组件并评估 ≥7GiB replay 需求。
- [ ] 设计层次 RL 与 Go-Explore 所需的环境扩展（savestate/目标接口）。
- [ ] 收集 Dreamer/MuZero 参考实现，评估投入产出。

> 文档将随实验推进更新；引用外部论文/实现时请在后续版本补充来源与结果对比。
