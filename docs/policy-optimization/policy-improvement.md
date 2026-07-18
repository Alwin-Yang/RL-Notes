# Policy Improvement

## REINFORCE vs Reward-to-go

| | REINFORCE | Reward-to-go (Causality trick) |
|---|---|---|
| 公式 | $$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\left(\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})\right)\left(\sum_{t=1}^{T} r(\mathbf{s}_{i,t},\mathbf{a}_{i,t})\right)$$ | $$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})\left(\sum_{t'=t}^{T} r(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})\right)$$ |
| 权重项 | 整条轨迹的**总回报**（从 t=1 到 T），对所有时间步都一样 | 每个时间步 t 对应**不同的**权重，只算从 t 到 T 的未来奖励 |
| 是否用到未来之前的奖励 | 用到，不管发生在 t 之前还是之后 | 不用，只用 t 及之后的奖励 |
| 方差 | 较高 | 较低（理论上无偏，但方差更小） |
| 直觉类比 | 团队赢了平分奖金，不管谁参与得早晚 | 只给该动作之后实际产生的奖励"记功" |

## 关键区别一目了然

两者的核心差别在括号的位置和范围：原始版本把 log π 求和整体括起来乘以**一个共同的**总回报，而 reward-to-go 版本把求和拆开，给每个时间步单独配上从该时刻起算的**局部**未来回报。

这个改动利用了因果性——即 t 时刻的动作不可能影响 t 之前已经发生的奖励，因此把这部分"无关"的历史奖励项去掉，数学上可以证明梯度的期望值不变（仍是无偏估计），但方差显著降低。


## Introducing a baseline
