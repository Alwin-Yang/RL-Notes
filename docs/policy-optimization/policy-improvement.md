# Policy Improvement

---
## REINFORCE vs Reward-to-go

### REINFORCE

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N}
\left(\sum_{t=1}^{T}\nabla_\theta
\log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})\right)
\left(\sum_{t=1}^{T}r(\mathbf{s}_{i,t},\mathbf{a}_{i,t})\right)
$$

- **Weight:** The total return of the entire trajectory is applied to every time step.
- **Past rewards:** Rewards received before the action are included.
- **Variance:** Higher.
- **Intuition:** The whole team shares the reward for winning, regardless of when each member contributed.

### Reward-to-go (Causality trick)

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})
\left(\sum_{t'=t}^{T}r(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})\right)
$$

- **Weight:** Each time step uses its own future return from $t$ to $T$.
- **Past rewards:** Rewards received before the action are excluded.
- **Variance:** Lower while the estimator remains unbiased.
- **Intuition:** Only rewards received after an action are credited to that action.

## Key difference

The key difference is the placement and scope of the summations. REINFORCE multiplies the sum of all log-policy gradients by one shared total return. Reward-to-go instead assigns each time step its own future return starting from that time step.

This change uses causality: an action at time $t$ cannot affect rewards received before $t$. Removing these unrelated past rewards leaves the expected gradient unchanged, so the estimator remains unbiased, while reducing its variance.


---
## Introducing A Baseline

Reward-to-go removes rewards that an action could not have caused, but the remaining return can still vary substantially across trajectories. We can further reduce this variance by subtracting a baseline $b(s_{i,t})$:

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})
\left(G_{i,t}-b(\mathbf{s}_{i,t})\right),
$$

where $G_{i,t}=\sum_{t'=t}^{T}r(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})$ is the reward-to-go.

Subtracting $b(\mathbf{s}_{i,t})$ does not change the expected policy gradient. Conditioned on a state $s$, the expected gradient contribution of the baseline is zero:

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)b(s)\right]
&=b(s)\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)\\
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

Therefore, the $-b(s)$ term is valid: its expectation is zero, so subtracting it preserves unbiasedness. Adding it would also preserve unbiasedness, but subtraction with a well-chosen baseline centers the return and reduces variance.

A common choice is $b(s_t)\approx V^\pi(s_t)$. The weight then becomes

$$
G_t-V^\pi(s_t)\approx A^\pi(s_t,a_t).
$$

The policy therefore increases the probability of actions that perform better than expected and decreases the probability of actions that perform worse than expected. The baseline may depend on the state, but it must not depend directly on the sampled action unless an appropriate correction is applied; otherwise the zero-expectation argument generally no longer holds.

!!! note
    A baseline reduces variance but does not eliminate sampling noise. Policy-gradient estimates can still be noisy and high-variance, especially with sparse or delayed rewards. In practice, vanilla Monte Carlo policy gradients work best with dense, informative rewards and large rollout batches.


---
## Using Surrogate Objectives

The policy-gradient estimator above already tells us the desired parameter
gradient:

### Policy-gradient estimator

$$
\widehat{g}
=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
\left(G_{i,t}-b(\mathbf{s}_{i,t})\right).
$$

Here $G_{i,t}$ is the reward-to-go. This expression is a vector: it specifies
the gradient that should be used to update the policy parameters. In principle,
one could compute every score vector
$\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})$ and
form the weighted sum explicitly. That is unnecessary, however, because an
automatic-differentiation system can obtain the same sum from a single scalar
objective.

### Surrogate objective

$$
\widehat{J}_{\mathrm{surr}}(\theta)
=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
\operatorname{stopgrad}\!\left(G_{i,t}-b(\mathbf{s}_{i,t})\right).
$$

For a fixed batch of sampled trajectories and fixed return-minus-baseline weights,

$$
\nabla_\theta\widehat{J}_{\mathrm{surr}}(\theta)=\widehat{g}.
$$

The surrogate objective is therefore a convenient scalar whose gradient
matches the sampled policy-gradient estimate. It is *surrogate* because its
numerical value is not itself the expected return $J(\theta)$; only its local
gradient is being used for the policy update.

The `stopgrad` operation, written as `detach()` in PyTorch, treats the sampled
return minus the baseline as a constant weight during the actor update.
This is especially important when the baseline is produced by a learned value
network: the actor loss should not update the critic through
$b(\mathbf{s}_{i,t})$. The critic is instead trained with its own value loss.

---
---

## 中文版本

---
### REINFORCE 与 Reward-to-go

#### REINFORCE

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N}
\left(\sum_{t=1}^{T}\nabla_\theta
\log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})\right)
\left(\sum_{t=1}^{T}r(\mathbf{s}_{i,t},\mathbf{a}_{i,t})\right)
$$

- **权重项：**整条轨迹的总回报，对所有时间步都相同。
- **历史奖励：**会使用动作发生之前的奖励。
- **方差：**较高。
- **直觉：**团队赢了之后平分奖金，不区分每个人何时作出贡献。

#### Reward-to-go（因果性技巧）

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})
\left(\sum_{t'=t}^{T}r(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})\right)
$$

- **权重项：**每个时间步使用各自从 $t$ 到 $T$ 的未来回报。
- **历史奖励：**不使用动作发生之前的奖励。
- **方差：**更低，同时仍是无偏估计。
- **直觉：**只把动作发生之后得到的奖励记作该动作的贡献。

### 关键区别

两者的核心差别在求和符号的位置和范围：REINFORCE 把所有 log-policy gradient 相加后，乘以一个共同的轨迹总回报；reward-to-go 则为每个时间步分别配上从该时刻开始的未来回报。

这个改动利用了因果性：$t$ 时刻的动作不可能影响 $t$ 之前已经发生的奖励。去掉这些无关的历史奖励不会改变梯度的期望，因此估计仍然无偏，但方差更低。

---
## 引入基线

Reward-to-go 去掉了动作不可能影响的历史奖励，但剩余回报在不同轨迹之间仍可能有很大波动。可以进一步减去 baseline $b(s_{i,t})$ 来降低方差：

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}\vert\mathbf{s}_{i,t})
\left(G_{i,t}-b(\mathbf{s}_{i,t})\right),
$$

其中 $G_{i,t}=\sum_{t'=t}^{T}r(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})$ 是 reward-to-go。

减去 $b(\mathbf{s}_{i,t})$ 不会改变 policy gradient 的期望。给定状态 $s$，baseline 对梯度的期望贡献为 0：

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)b(s)\right]
&=b(s)\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)\\
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

因此，写成 $-b(s)$ 是正确的：baseline 项的期望为 0，减去它不会引入偏差。加上它同样不会引入偏差，但使用减法并选择合适的 baseline，可以让回报以其平均水平为中心，从而降低方差。

常见的选择是 $b(s_t)\approx V^\pi(s_t)$。此时权重变为

$$
G_t-V^\pi(s_t)\approx A^\pi(s_t,a_t).
$$

因此，policy 会提高表现好于预期的动作概率，并降低表现差于预期的动作概率。Baseline 可以依赖状态，但不能直接依赖采样得到的动作；否则在没有额外修正时，上述期望为 0 的推导通常不再成立。

!!! note "注意"
    Baseline 只能降低方差，不能消除采样噪声。Policy-gradient 估计仍可能具有较大噪声和较高方差，特别是在奖励稀疏或延迟的任务中。实践中，基础的 Monte Carlo policy gradient 通常在奖励密集且信息充分、rollout batch 较大时表现更好。


---
## 使用代理目标

上面的 policy-gradient estimator 已经给出了需要用来更新 policy 参数的梯度：

### Policy-gradient estimator

$$
\widehat{g}
=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
\left(G_{i,t}-b(\mathbf{s}_{i,t})\right).
$$

其中 $G_{i,t}$ 是 reward-to-go。这个表达式是一个向量，表示更新 policy
参数时需要使用的梯度。原则上，可以分别计算每一个 score vector
$\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})$，再对它们
进行加权求和。但没有必要逐项计算，因为自动微分框架可以通过一个标量目标，
一次得到相同的梯度和。

### 代理目标

$$
\widehat{J}_{\mathrm{surr}}(\theta)
=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
\operatorname{stopgrad}\!\left(G_{i,t}-b(\mathbf{s}_{i,t})\right).
$$

对于一批固定的采样轨迹以及固定的“回报减 baseline”权重，有

$$
\nabla_\theta\widehat{J}_{\mathrm{surr}}(\theta)=\widehat{g}.
$$

因此，代理目标是一个便于自动微分计算的标量，其梯度与采样得到的 policy-gradient
estimator 相同。之所以称为“代理”目标，是因为它的数值本身并不是期望回报
$J(\theta)$；在 policy 更新中真正使用的是它的局部梯度。

`stopgrad` 在 PyTorch 中写作 `detach()`。它使自动微分系统在 actor 更新期间，
把采样回报减去 baseline 后的结果视为常数权重。当 baseline 由一个可学习的
value network 给出时，这一点尤其重要：actor 的目标不应该通过
$b(\mathbf{s}_{i,t})$ 更新 critic。Critic 应当通过自己的 value loss 单独训练。
