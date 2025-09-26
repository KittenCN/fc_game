# FC Emulator Toolkit

该项目提供一个基于 Python 的 FC (NES) 模拟器工具集，方便加载 `.nes` ROM、手动游玩，同时为强化学习自动化训练提供接口。核心基于 [`nes-py`](https://github.com/Kautenja/nes-py)，并额外封装了控制器输入、图像渲染、环境交互等模块，使后续开发智能体更高效。

## 功能概览
- 加载 iNES 格式的 ROM 并运行。
- 使用键盘或自定义输入器进行实时游玩。
- 导出屏幕像素、内存状态等观测信息。
- 提供 Gymnasium 兼容的环境包装器，方便训练/推理神经网络。
- 集成 Stable-Baselines3 训练脚本，可一键启动 PPO/A2C 训练。

## 快速开始

```bash
pip install -e .
python -m fc_emulator.cli --rom path/to/game.nes
```

默认控制键位：
- 方向键：`WASD`
- `A` 键：`J`
- `B` 键：`K`
- `Start`：`Enter`
- `Select`：`Right Shift`

## 强化学习训练与推理
1. 安装额外依赖：
   ```bash
   pip install -e .[rl]
   ```
   若之前已安装过，可单独补充图像缩放依赖：`pip install pillow`。
2. 使用内置训练脚本（默认使用离散化动作集合与灰度观测）：
   ```bash
   python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
       --algo ppo --total-timesteps 1000000 --tensorboard
   ```
   - `--action-set` 可选择 `default`/`simple`，或自定义按键组合，例如：`"RIGHT;A,RIGHT;B"`。
   - `--frame-skip` / `--frame-stack` 控制子采样与状态堆叠，兼容 Stable-Baselines3 的 `CnnPolicy`。
   - `--resize HEIGHT WIDTH` 可以在进入策略网络前对观测图像下采样（如 `--resize 84 84`），显著降低计算量并加速训练。
   - `--vec-env subproc` 强制使用多进程采样，使 CPU 多核更充分，默认 `auto` 会在 `--num-envs>1` 时自动启用。
   - 默认会在 reset 后自动按下 START 按钮跳过标题界面，可用 --disable-auto-start / --auto-start-max-frames / --auto-start-press-frames 调整
   - ����ͣ�ͼ�⣺Ĭ���ڻ����ƶ� 900 ��֡����（约 15 秒）�Լ� 1 �����ƶȽ��о������ѵ����� --stagnation-frames / --stagnation-progress ����
   - 支持 ε-greedy 探索，可通过 --exploration-epsilon / --exploration-final-epsilon / --exploration-decay-steps 调整随机动作概率
   - 增加 `--num-envs` 可以并行运行多个环境，进一步压榨 CPU/GPU 吞吐。
3. 载入已训练模型并实时推理：
   ```bash
   python -m fc_emulator.infer --rom roms/SuperMarioBros.nes \
       --model runs/ppo_agent_20250101-120000.zip --deterministic
   ```
   推理时会自动开启 `render_mode="human"`，可观察智能体操作过程。若训练时使用了下采样，请传入同样的 `--resize` 参数保持输入尺寸一致。

## 面向自动化训练
- `fc_emulator.rl_env.NESGymEnv` 基于 `gymnasium.Env`，暴露 `step/reset/render` 接口。
- `fc_emulator.wrappers.DiscreteActionWrapper` 将 8 个按键映射到可配置的有限动作集合。
- `fc_emulator.train` / `fc_emulator.infer` 封装了 Stable-Baselines3 的向量环境、帧堆叠与模型保存/加载流程。
- 可通过自定义回调、奖励函数或观察空间扩展，快速接入其它算法库。

## 项目结构
- `fc_emulator/`
  - `rom.py`: ROM 解析与元数据。
  - `bus.py`: 内存映射与设备总线。
  - `controller.py`: 控制器输入映射。
  - `renderer.py`: Pygame 渲染和窗口管理。
  - `emulator.py`: 对 `nes_py.NESEnv` 的高级封装。
  - `rl_env.py`: Gymnasium 环境包装器。
  - `wrappers.py`: 强化学习动作集合与离散化封装。
  - `rl_utils.py`: 训练/推理共用的环境构建工具。
  - `train.py`: 训练 CLI（PPO/A2C）。
  - `infer.py`: 推理 CLI。
  - `cli.py`: 命令行入口，支持人类游玩。
- `examples/`
  - `human_play.py`: 手动游玩的示例。
  - `random_agent.py`: 随机策略演示。

## TODO
- 保存/读取状态（save state）。
- 网络联机对战。
- 更精细的帧率同步与音频输出。
- 增加更多动作预设与训练曲线可视化工具。

欢迎根据需要扩展，用于神经网络训练或其它自动化项目。
