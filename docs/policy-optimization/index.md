# Policy Optimization

Policy optimization directly adjusts the parameters of a policy to increase
expected return. This page derives the policy gradient, which is the foundation
for the methods in this section.

## Define the policy objective

Let $\pi_\theta$ be a policy with parameters $\theta$. A trajectory $\tau$ is
sampled from the distribution $p_\theta$ induced by that policy and the
environment. For a trajectory of length $T$, the objective is

$$
J(\theta)
=\mathbb{E}_{\tau \sim p_\theta}
\left[\sum_{t=1}^{T}r(s_t,a_t)\right].
$$

Here, $r(s_t,a_t)$ is the reward received after taking action $a_t$ in state
$s_t$. The objective is the expected return under the current policy. Policy
optimization therefore performs gradient ascent on $J(\theta)$.

## Turn the derivative into an expectation

Writing the expectation as an integral gives

$$
J(\theta)
=\int p_\theta(\tau)
\left(\sum_{t=1}^{T}r(s_t,a_t)\right)d\tau.
$$

Assuming the reward function does not depend directly on $\theta$,
differentiate the trajectory distribution:

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\int \nabla_\theta p_\theta(\tau)
\left(\sum_{t=1}^{T}r(s_t,a_t)\right)d\tau\\
&=\int p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
\left(\sum_{t=1}^{T}r(s_t,a_t)\right)d\tau\\
&=\mathbb{E}_{\tau\sim p_\theta}
\left[\nabla_\theta\log p_\theta(\tau)
\left(\sum_{t=1}^{T}r(s_t,a_t)\right)\right].
\end{aligned}
$$

The second line uses the **log-derivative identity**, also called the
**score-function identity**. For any sample $x$ with $p_\theta(x)>0$, apply the
chain rule to the logarithm:

$$
\nabla_\theta\log p_\theta(x)
=\frac{1}{p_\theta(x)}\nabla_\theta p_\theta(x).
$$

Multiplying both sides by $p_\theta(x)$ gives

$$
\boxed{
\nabla_\theta p_\theta(x)
=p_\theta(x)\nabla_\theta\log p_\theta(x)
}.
$$

This identity replaces the derivative of a probability distribution with the
distribution itself multiplied by a log-probability gradient. An integral of
the form

$$
\int \nabla_\theta p_\theta(x)f(x)\,dx
$$

can therefore be rewritten as an expectation:

$$
\begin{aligned}
\int \nabla_\theta p_\theta(x)f(x)\,dx
&=\int p_\theta(x)
\nabla_\theta\log p_\theta(x)f(x)\,dx\\
&=\mathbb{E}_{x\sim p_\theta}
\left[\nabla_\theta\log p_\theta(x)f(x)\right].
\end{aligned}
$$

The expectation can then be estimated using samples from $p_\theta$, even when
the integral cannot be evaluated analytically. In policy optimization, $x$ is
an entire trajectory, $p_\theta(x)$ is its probability under the current
policy, and $f(x)$ is the trajectory's reward sum.

## Remove the environment dynamics

For a trajectory $\tau=(s_1,a_1,\ldots,s_T,a_T,s_{T+1})$,

$$
p_\theta(\tau)
=\rho_0(s_1)\prod_{t=1}^{T}
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

The initial-state distribution $\rho_0$ and environment transition distribution
$P$ do not depend on the policy parameters. Their gradients are therefore zero,
leaving only the policy terms:

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=1}^{T}\nabla_\theta
\log\pi_\theta(a_t\mid s_t).
$$

Substituting this result into the objective gradient gives the trajectory-level
policy-gradient formula:

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\tau\sim p_\theta}
\left[
\left(\sum_{t=1}^{T}\nabla_\theta
\log\pi_\theta(a_t\mid s_t)\right)
\left(\sum_{t'=1}^{T}r(s_{t'},a_{t'})\right)
\right].
$$

## Estimate the gradient from trajectories

Because the expectation cannot usually be computed exactly, REINFORCE
estimates it with $N$ sampled trajectories:

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N}
\left(\sum_{t=1}^{T}\nabla_\theta
\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})\right)
\left(\sum_{t'=1}^{T}r(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})\right).
$$

This estimate is unbiased but can have high variance. Both the policy and the
environment may be stochastic, so different trajectories can produce very
different returns.

## Where to go next

### Policy improvement

[Policy Improvement](policy-improvement.md) explains how reward-to-go,
baselines, and surrogate objectives turn the basic estimator into a more useful
training objective.

### Off-policy learning

[Off-policy Learning](off-policy.md) explains why data from another policy
requires a correction and derives importance sampling. This provides the
foundation for understanding how off-policy actor-critic methods reuse data.

### Actor-critic

[Actor-Critic](actor-critic.md) replaces the sampled return or baseline
with a learned value estimate. It also introduces advantage estimation, TD
targets, $n$-step returns, and GAE.

The algorithms in this family are:

- [A2C and A3C](a2c-a3c.md): direct on-policy actor-critic
  methods.
- [TRPO](trpo.md): limits the actor update with a KL-divergence
  constraint.
- [PPO](ppo.md): simplifies TRPO's idea with a clipped surrogate
  objective.
- [SAC](../actor-critic/sac.md): learns off-policy critics and includes an
  entropy objective.

## Questions

See the [Policy Optimization Q&A](q-and-a.md) for conceptual questions about
gradient ascent, gradient variance, rollout batches, and PPO clipping.
