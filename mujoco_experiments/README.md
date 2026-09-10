# MuJoCo 笛卡尔阻抗实验（与 ROS 仓库解耦）

本目录是独立 Python 工程：不编译、不链接、不 import `cartesian_impedance_controller`。
机械臂模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 的 Franka Panda，
执行器已改成力矩 `motor`。

控制律（实验 1 关闭零空间和外力前馈）：

```text
tau = J^T (-K e - D J qdot) + qfrc_bias
```

`qfrc_bias` 补偿 MuJoCo 里的重力与科氏力。参数全部在 `config/*.yaml`。

## 环境

```bash
conda activate gct   # 已安装 mujoco 3.12 时
# 或: python -m pip install -r mujoco_experiments/requirements.txt
```

## 实验 1：固定参考 + 一次冲量

末端先在 home 位姿锁住参考。`impulse_time` 时在 `attachment_site` 打一次 6 维扳手冲量：
`impulse` 是力 [N·s]，`angular_impulse` 是力矩 [N·m·s]，都只持续一个仿真步。
位置和姿态都会被推偏，再靠阻抗拉回。窗口里红箭是力、蓝箭是力矩。

```bash
cd mujoco_experiments
python scripts/run_experiment1.py --config config/experiment1.yaml
```

默认会打开 MuJoCo 窗口（约 8 秒，按物理时间播放）：绿球是锁住的参考位姿，红点是末端，施力时出现红箭。只要曲线、不要窗口时加 `--headless`，或把 yaml 里 `sim.viewer` 设为 `false`。

主要可调项见 `config/experiment1.yaml`：

- `impedance.stiffness_trans`：三轴线刚度 [N/m]
- `impedance.damping_zeta`：相对阻尼；或直接写 `damping_trans` / `damping_rot`
- `experiment.impulse`：一次冲量 [N·s]
- `experiment.impulse_time`：打冲量的时刻 [s]

日志：`logs/experiment1.csv`、`logs/experiment1.png`、`logs/experiment1_torque.png`（七个关节命令力矩）。开窗口运行时，仿真结束后会弹出位移和力矩两张图。

## 实验 2：期望位姿大阶跃 + 有无滤波

1 秒时把期望位置和姿态一起阶跃到远处（默认约 30 cm、几十度，不是小抖动）。
`filtering.pose` 与仓库相同：`1.0` 关闭滤波，误差瞬间变大、力矩会跳；`0.1` 则先把参考平滑过去。
`filtering.compare: true` 时会再静默跑一遍无滤波，两张图叠在一起。

```bash
python scripts/run_experiment2.py --config config/experiment2.yaml
```

窗口：黄球是阶跃后的期望，绿球是滤波后的参考，红点是末端。关滤波时绿球会跟着黄球一起跳。结束后弹出误差、末端位置、关节力矩、末端加速度曲线。

## 实验 3：期望位置设进墙里

前方有固定半透明刚墙。1 秒后把期望位置沿 x 设到墙内。末端应停在墙外。结束后画出**末端力/力矩**（命令扳手 fx–fz、mx–mz）。

`control.mode` 选 `impedance` 或 `position`（高刚度笛卡尔 PD，当作纯位置）。`control.compare: true` 时两种都跑，虚线是另一种。

```bash
python scripts/run_experiment3.py --config config/experiment3.yaml
```

只看位置控制：把 `mode` 改成 `position`。曲线：`logs/experiment3_wrench.png`。

## 实验 4：锁死末端，改零空间构型

无墙。笛卡尔期望一直锁在初始末端位姿。1 秒时把零空间关节目标阶跃一截（`experiment.q_offset`）。
投影后肘部等冗余关节应动，红点应几乎不离开绿球。

```bash
python scripts/run_experiment4.py --config config/experiment4.yaml
```

曲线：`logs/experiment4_ee.png`（末端误差）、`logs/experiment4_q.png`（关节角；虚线是零空间目标，不能全部达到）。

## 实验 5：墙上接触 + Z 向前馈力

保留刚墙。1 秒时期望伸进墙内；同时 `Kz=0.01`，世界系 `Fz=10 N` 前馈（`τ += Jᵀ w`）。看墙上作用在机器人上的接触力。

```bash
python scripts/run_experiment5.py --config config/experiment5.yaml
```

曲线：`logs/experiment5_contact.png`（Fx/Fy/Fz 与合力；Fz 虚线是 10 N 前馈）。+z 向上；要往下压把 `feedforward.force` 的 z 改成 `-10`。
