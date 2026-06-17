下面我直接为你生成一份 **可粘贴到仓库中的 Markdown 文档**：
`docs/beyondmimic_classic_methods.md`

格式专业、内容全面、适合放进你的项目文档库，
并且完全针对 **Humanoid Tracking / BeyondMimic / IsaacLab / RSL-RL** 这种任务设计。

---

# 📘 BeyondMimic & Tracking Task —— 经典训练思路全集

（For IsaacLab + RSL-RL + Humanoid Tracking）

本文件收集了在 **人体动作追踪、模仿学习、强化学习（DeepMimic / AMP / BeyondMimic）** 中反复验证有效的经典工程技巧。
适用于：

* Tracking（运动跟踪）
* RL-Based Control（强化学习控制）
* Humanoid MPC/RL 混合控制
* Sim2Real 训练
* Locomotion / Freestyle / Skill Learning

这些技巧可用于提升：

* 稳定性
* 收敛速度
* 动作自然性
* 模型鲁棒性
* Sim2Real 性能

---

# 🟥 1. Curriculum Learning（课程训练策略）

---

## **1. 动态误差课程（Error-based Curriculum）**

根据 tracking 误差自动调整难度：

* 误差小 → 提高随机化 & 外力强度
* 误差大 → 降低难度

**优点：** 自动、自适应、更稳。

---

## **2. Phase Curriculum（短 → 长动作）**

1. 先训 0.5 秒片段
2. 再训 1 秒
3. 再 2 秒
4. 最后整段 motion

**适合：舞蹈 / 跳跃 / 长动作Tracking**

---

## **3. 慢速 → 正常速度 Curriculum**

参考动作从低速播放到 100%：

* 0.5× → 0.8× → 1.0×

减少 early collapse。

---

# 🟦 2. Observation & Action 设计技巧

---

## **4. Relative Observation（相对观测）**

所有坐标转换到 root frame：

* 更容易学习
* Sim2Real 更鲁棒
* 与运动速度无关

Tracking 必备。

---

## **5. Residual Action（增量动作）**

将动作拆成：

```
final_action = reference_action + residual
```

Residual 范围很小，使 tracking 稳定性极大提升。

---

# 🟩 3. Reward Engineering 经典方法

---

## **6. Hierarchical Reward（层级奖励）**

奖励分三层：

* **低级：**姿态/速度/关节
* **中级：**接触、平衡、foot clearance
* **高级：**节奏、smoothness、风格

结构化奖励更好调。

---

## **7. Sparse + Dense Reward 混合**

Dense tracking reward + Sparse 成就奖励：
例如完成跳跃/turn 时给 bonus。

---

## **8. Smoothness Reward（平滑奖励）**

```
r_smooth = exp(- α * |qdd|)
```

控制关节加加速度 → 动作更像真人。

---

# 🟨 4. Dynamics Robustness（动态鲁棒性）

---

## **9. 状态扰动（State Perturbations）**

在 obs 中加入噪声：

* 角度噪声
* 速度噪声
* 地面高度噪声

Sim2Real 效果极好。

---

## **10. Action Dropout / Delay**

模拟真实机器人执行延迟：

```
if rand < p:
    action = last_action
```

让策略在延迟下依然稳定。

---

## **11. Trajectory Noise Augmentation**

对 reference motion 加：

* jitter
* phase shift
* amplitude scaling

Tracking 任务鲁棒性提升明显。

---

# 🟧 5. Training Framework 设计技巧

---

## **12. Two-Stage Training（双阶段训练）**

Stage 1
只学 pose/vel tracking，不加复杂奖励
→ 稳定学习基础动作

Stage 2
加入 contact、balance、foot clearance
→ 学真实且自然的动作

---

## **13. Value Bootstrapping（值函数热启动）**

先训一个小网络的价值函数
再迁移到大网络 → 避免 PPO 初期崩溃。

---

## **14. Replay Buffer Hybrid IL（离线+在线混合模仿）**

IL 数据在 replay buffer 中混合 Online rollout：

* 稳定
* 容易模仿大动作
* 初期不容易炸

---

# 🟫 6. Motion 修复/重建 技巧

---

## **15. Phase Estimation Network（相位估计）**

加入一个 φ（动作相位）作为 obs：
tracking 动作稳如狗。

---

## **16. Motion Blending（动作平滑过渡）**

动作切换时：

```
q = w * q_a + (1-w) * q_b
```

避免姿态跳变。

---

## **17. Adaptive Target Scaling（自适应跟踪权重）**

对于 tracking 难的 body：

* early stage：低权重
* later stage：高权重

这是 “reward curriculum” 的变种。

---

# 🟩 总结（可放封面）

| 类别                 | 技巧数量 | 适用任务                    |
| ------------------ | ---- | ----------------------- |
| Curriculum（课程）     | 3    | Tracking / Locomotion   |
| Obs & Action       | 2    | 所有 Humanoid RL          |
| Reward Engineering | 3    | DeepMimic / BeyondMimic |
| Dynamic Robustness | 3    | Sim2Real                |
| Training Framework | 3    | 大规模 RL                  |
| Motion Repair      | 3    | Tracking / Freestyle    |

总计 **17 条经典策略**。

---

# 🧩 附：推荐结合方式（实践最佳流程）

你项目中最推荐这样组合：

### ✔ Residual Action

### ✔ Phase Curriculum（短→长动作）

### ✔ Slow→Fast Curriculum

### ✔ 状态扰动 + 轻量 domain rand

### ✔ Two-Stage Training

### ✔ Smoothness Reward

### ✔ Phase Estimation（如有复杂动作）


