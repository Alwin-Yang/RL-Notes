# Proximal Policy Optimization

Proximal policy optimization (PPO) is an on-policy actor-critic algorithm. Like
[TRPO](trpo.md), it prevents one batch of policy updates from moving the actor
too far from the policy that collected the batch.

## Review: actor-critic with multiple actor updates

The basic
[actor-critic walkthrough](actor-critic.md#a-full-algorithm-walkthrough)
trains two networks side by side: the actor $\pi_\theta(a\mid s)$ and the critic
$V_\phi^\pi(s)$. To reuse one batch for multiple actor updates, keep its first
three steps and change steps 4 and 5:

1. **Collect data.** Set $\theta_{\mathrm{old}}\leftarrow\theta$, run
   $\pi_{\theta_{\mathrm{old}}}$ in the environment, and store a batch of
   trajectories
   $\{(\mathbf{s}_{i,1},\mathbf{a}_{i,1},\ldots,
   \mathbf{s}_{i,T},\mathbf{a}_{i,T})\}_{i=1}^{N}$.
2. **Fit the critic.** Build a target $y_{i,t}$ for every visited step, then
   take a few gradient steps on
   $\mathcal{L}(\phi)=\frac{1}{2N}\sum_{i,t}
   (V_\phi^\pi(\mathbf{s}_{i,t})-y_{i,t})^2$.
3. **Evaluate advantages.** For all $i$ and $t$, compute
   $\widehat{A}_{i,t}
   =r_{i,t}+\gamma V_\phi^\pi(\mathbf{s}_{i,t+1})
   -V_\phi^\pi(\mathbf{s}_{i,t})$, or its $n$-step or GAE version.
4. **Estimate the policy gradient with importance sampling.** Reweight every
   sampled action by the fraction
   $\frac{\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})}
   {\pi_{\theta_{\mathrm{old}}}
   (\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})}$:

    $$
    \nabla_\theta J(\theta)
    \approx
    \frac{1}{N}
    \sum_{i,t}
    \frac{
    \pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
    }{
    \pi_{\theta_{\mathrm{old}}}
    (\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
    }
    \nabla_\theta
    \log\pi_\theta
    (\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
    \widehat{A}_{i,t}.
    $$

5. **Update the actor multiple times.** Apply
   $\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)$ for $K$ gradient
   steps on the same batch, recomputing the probability fraction after every
   step. Then throw the batch away and go back to step 1.

The fraction is one before the first actor update. Later updates move
$\pi_\theta$ away from $\pi_{\theta_{\mathrm{old}}}$, so some fractions can
become very large or very small. PPO adds clipping to prevent those samples
from driving further policy changes.

!!! note "Why PPO is still called on-policy"
    After the first actor update, $\pi_\theta$ differs from the behavior policy
    $\pi_{\theta_{\mathrm{old}}}$. Later gradient steps therefore use data that
    is technically off-policy relative to the current actor. PPO is still
    classified as on-policy because it collects a fresh batch from the latest
    policy each iteration, reuses that batch only for a limited number of
    updates, and then discards it. The probability ratio corrects the sampled
    action probabilities, while clipping keeps the two policies close; it does
    not make PPO a replay-buffer-based off-policy algorithm such as SAC.

## The surrogate objective

Step 4 specifies the policy gradient as a vector, but an automatic
differentiation system works most conveniently from a scalar objective.
Instead of constructing and summing the score vectors by hand, find a scalar
whose gradient equals that estimate. The convenient log-derivative identity is

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)\nabla_\theta\log p_\theta(x).
$$

The old policy and the advantage estimates are fixed during the actor update.
Applying the identity to each term gives

$$
\begin{aligned}
\frac{
\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}{
\pi_{\theta_{\mathrm{old}}}
(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}
\nabla_\theta\log\pi_\theta
(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
\widehat{A}_{i,t}
&=
\frac{
\nabla_\theta
\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}{
\pi_{\theta_{\mathrm{old}}}
(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}
\widehat{A}_{i,t}
\\
&=
\nabla_\theta
\left[
\frac{
\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}{
\pi_{\theta_{\mathrm{old}}}
(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}
\widehat{A}_{i,t}
\right].
\end{aligned}
$$

The entire gradient estimate can therefore be obtained by differentiating the
scalar surrogate objective

$$
\widetilde{J}(\theta)
=
\frac{1}{N}
\sum_{i,t}
\frac{
\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}{
\pi_{\theta_{\mathrm{old}}}
(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
}
\widehat{A}_{i,t}.
$$

The actor maximizes $\widetilde{J}(\theta)$ with automatic differentiation
instead of constructing the policy gradient by hand.

## Trick 1: Clip the importance weights

For compactness, denote the probability fraction by

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

It is greater than one when an update makes the sampled action more likely and
less than one when it makes the action less likely. Repeated updates can push
it far from one.

A first solution is to clip the importance weight:

$$
\widetilde{J}^{\mathrm{clip-only}}(\theta)
=
\mathbb{E}_t
\left[
\operatorname{clip}
\left(\rho_t(\theta),1-\epsilon,1+\epsilon\right)\widehat{A}_t
\right],
$$

where $\widehat{A}_t$ is the critic's advantage estimate and $\epsilon$ is the
clipping range. Once $\rho_t(\theta)$ leaves this interval, the clipped term is
constant, so it no longer rewards moving the new policy farther from the
behavior policy.

## Trick 2: Take the minimum

Clipping alone has a failure case: it can replace the original term with a
larger value. Because the actor maximizes the objective, that would make a
harmful policy change look better. PPO therefore takes the minimum of the
original and clipped terms:

$$
\widetilde{J}^{\mathrm{PPO}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
\rho_t(\theta)\widehat{A}_t,
\operatorname{clip}
\left(\rho_t(\theta),1-\epsilon,1+\epsilon\right)\widehat{A}_t
\right)
\right].
$$

The minimum makes the objective pessimistic:

- If $\widehat{A}_t>0$, increasing the action probability helps, but the
  improvement is capped once $\rho_t(\theta)>1+\epsilon$.
- If $\widehat{A}_t<0$, decreasing the action probability helps, but the
  improvement is capped once $\rho_t(\theta)<1-\epsilon$.

On the other side of the interval, PPO keeps the original term instead of
letting clipping make the objective look better. This is the final PPO
surrogate objective.

Clipping is not a hard constraint. Other samples can still move the policy, so
implementations often monitor the KL divergence and stop an update early when
it becomes too large.

## Relationship to KL divergence

### PPO-Clip

PPO-Clip does not include a KL-divergence term in its objective. Its main
mechanism is probability-ratio clipping. An implementation may still monitor
the sample estimate

$$
\widehat{D}_{\mathrm{KL}}
=
\mathbb{E}_t
\left[
\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
-
\log\pi_\theta(a_t\mid s_t)
\right]
$$

and stop the remaining updates when it exceeds a target value. The expectation
is the average over the current batch.

### PPO-Penalty

PPO-Penalty adds a KL penalty to the surrogate objective:

$$
\widetilde{J}^{\mathrm{KL}}(\theta)
=
\mathbb{E}_t
\left[
\rho_t(\theta)\widehat{A}_t
-
\beta
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\middle\|
\pi_\theta(\cdot\mid s_t)
\right)
\right],
$$

where $\beta>0$ is the penalty coefficient. The algorithm increases $\beta$
when the observed KL divergence is too large and decreases it when the
divergence is too small.

### TRPO

[TRPO](trpo.md) instead treats the KL divergence as an explicit trust-region
constraint:

$$
\begin{aligned}
\max_\theta\quad
&\mathbb{E}_t
\left[
\rho_t(\theta)\widehat{A}_t
\right]
\\
\text{subject to}\quad
&\mathbb{E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\middle\|
\pi_\theta(\cdot\mid s_t)
\right)
\right]
\leq\delta,
\end{aligned}
$$

where $\delta>0$ is the maximum average KL divergence allowed in one policy
update.

## Trick 3: Generalized advantage estimation

The PPO objective still needs an advantage estimate $\widehat{A}_t$. First fit
the critic $V_\phi$ with a Monte Carlo or bootstrapped target. Then compute the
one-step temporal-difference residual

$$
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

Generalized advantage estimation (GAE) combines residuals over different
horizons:

$$
\widehat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{T-t}
(\gamma\lambda)^l\delta_{t+l},
$$

where $\lambda\in[0,1]$ controls how quickly longer-horizon terms lose weight.
Smaller $\lambda$ cuts the estimate earlier, reducing variance but relying more
on the critic and therefore introducing more bias. Larger $\lambda$ uses longer
returns, reducing critic bias but increasing variance.

See [Generalized advantage
estimation](actor-critic.md#generalized-advantage-estimation)
for the full derivation.

## Practical choices for repeated updates

Let $B$ be the number of transitions in one rollout batch, $M$ the mini-batch
size, and $K$ the number of optimization epochs. One PPO iteration takes
approximately

$$
K\frac{B}{M}
$$

actor and critic gradient steps before collecting new data.

### Recommended starting configurations

Use these as baselines before tuning one parameter at a time:

| Parameter | Continuous control | Atari-style input |
| --- | ---: | ---: |
| Environments $\times$ steps | $1\times2048$ | $8\times128$ |
| Rollout batch $B$ | $2048$ | $1024$ |
| Mini-batch size $M$ | $64$ | $256$ |
| Epochs $K$ | $10$ | $4$ |
| Learning rate | $3\times10^{-4}$ | $2.5\times10^{-4}$ |
| Policy clip $\epsilon$ | $0.2$ | $0.1$ |
| Discount $\gamma$ | $0.99$ | $0.99$ |
| GAE $\lambda$ | $0.95$ | $0.95$ |
| Advantage normalization | yes | yes |
| Entropy coefficient $c_H$ | $0$ | $0.01$ |
| Value-loss coefficient $c_V$ | $0.5$ | $0.5$ |
| Optional value clip $\epsilon_V$ | $0.2$ | $0.1$ |
| Maximum gradient norm | $0.5$ | $0.5$ |
| Optional target KL | $0.01$ | $0.01$ |

The continuous-control column follows the defaults used by
[Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
with one environment and by the
[CleanRL continuous-action
implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py).
The Atari column follows the
[CleanRL Atari implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py).
The target KL of $0.01$ is the early-stopping default in
[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

### Number of epochs

More epochs extract more optimization work from each environment transition,
but they also move $\pi_\theta$ farther from
$\pi_{\theta_{\mathrm{old}}}$. Choose $K$ together with the actor learning rate
and clipping range:

- Increase $K$ when the KL divergence and fraction of clipped samples remain
  small and the batch appears underused.
- Decrease $K$, reduce the actor learning rate, or stop early when the KL
  divergence grows too quickly.
- A high clip fraction means many samples have crossed the clipping boundary.
  Further epochs are then more likely to add policy drift than useful
  improvement.

For a continuous-control baseline with $K=10$, a practical first adjustment is
to reduce $K$ to $5$ when the approximate KL repeatedly exceeds its target
before all epochs finish. If the KL remains very small and the clip fraction is
low, compare $K=10$ with $K=15$ rather than assuming that more epochs will
help. An approximate-KL target of $0.01$, as used by
[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html),
is a conservative starting point for early stopping.

### Clipping range

The clipping range $\epsilon$ determines the interval
$[1-\epsilon,1+\epsilon]$. A practical search set is

$$
\epsilon\in\{0.1,0.2,0.3\}.
$$

Use $\epsilon=0.2$ as the general starting point:

- Use $\epsilon=0.1$ for a more conservative objective when the approximate KL
  grows too quickly. A narrower interval usually increases the measured clip
  fraction, so do not choose $\epsilon$ from clip fraction alone.
- Keep $\epsilon=0.2$ when training is stable and the actor makes measurable
  progress.
- Try $\epsilon=0.3$ only when the KL divergence and clip fraction stay low
  and policy learning appears too slow.

Tune $\epsilon$ together with the actor learning rate and $K$. A wider clipping
range, larger learning rate, and more epochs all allow the actor to move farther
on one rollout batch. Change one of them at a time so the cause of instability
remains identifiable.

For $\epsilon=0.2$, clipping starts at probability ratios $0.8$ and $1.2$.
This does not mean every action probability is restricted to a $20\%$ change.
The final objective takes a minimum with the unclipped term, and updates from
other samples can still push a ratio outside the interval. Clipping removes an
incentive; it is not a hard policy constraint.

### Mini-batch size

Smaller mini-batches produce noisier gradients and more optimizer steps per
epoch. Larger mini-batches produce smoother gradients but fewer parameter
updates. Shuffle the complete rollout batch before every epoch so consecutive
time steps do not remain grouped together.

The mini-batch must also contain enough positive- and negative-advantage samples
to give the actor a balanced update. Very small mini-batches can make one sign
dominate by chance.

### Learning rates and gradient clipping

Use a learning rate of $3\times10^{-4}$ as the first choice for a small MLP
actor-critic. A shared optimizer can use this rate for both networks. With
separate optimizers, keep the actor at $3\times10^{-4}$ and try
$10^{-3}$ for the critic when it learns too slowly.

- If the approximate KL grows too quickly, reduce the actor learning rate from
  $3\times10^{-4}$ to $10^{-4}$ before making several other changes.
- If the critic loss oscillates or grows, reduce the critic learning rate.
- Clip the combined gradient norm at $0.5$ as a default safeguard against an
  unusually large mini-batch update.

### Discount and GAE settings

Start with $\gamma=0.99$ and $\lambda=0.95$. Adjust them only when the reward
horizon or advantage quality gives a clear reason:

- For rewards delayed across very long horizons, compare $\gamma=0.99$ with
  $0.995$ or $0.999$.
- If the critic is reliable but the advantages remain too noisy, compare
  $\lambda=0.95$ with $0.9$.
- If critic bias is the larger problem and longer returns are affordable,
  compare $\lambda=0.95$ with $0.97$ or $0.99$.

Increasing $\gamma$ gives distant rewards more weight and changes the control
objective. Increasing $\lambda$ uses longer-horizon residuals, which reduces
reliance on the critic but usually increases variance. Tune them separately.

### Entropy bonus and value loss

When actor and critic losses are combined, maximize

$$
\widetilde{J}^{\mathrm{total}}
=
\widetilde{J}^{\mathrm{PPO}}
+c_H\mathbb{E}_t
\left[
\mathcal{H}
\left(\pi_\theta(\cdot\mid s_t)\right)
\right]
-c_V\mathcal{L}_V(\phi),
$$

where $c_H$ controls the entropy bonus and $c_V$ controls the critic loss.
Start with $c_V=0.5$. For continuous control, start with $c_H=0$ and try
$10^{-3}$ or $10^{-2}$ only if policy entropy collapses too early. For
Atari-style discrete control, $c_H=0.01$ is a reasonable starting point.

The ordinary critic loss is

$$
\mathcal{L}_V(\phi)
=
\frac{1}{2}
\mathbb{E}_t
\left[
\left(V_\phi(s_t)-y_t\right)^2
\right].
$$

An optional value-clipping variant first defines

$$
V_\phi^{\mathrm{clip}}(s_t)
=
V_{\phi_{\mathrm{old}}}(s_t)
+
\operatorname{clip}
\left(
V_\phi(s_t)-V_{\phi_{\mathrm{old}}}(s_t),
-\epsilon_V,
\epsilon_V
\right)
$$

and uses

$$
\mathcal{L}_V^{\mathrm{clip}}(\phi)
=
\frac{1}{2}
\mathbb{E}_t
\left[
\max\left(
\left(V_\phi(s_t)-y_t\right)^2,
\left(V_\phi^{\mathrm{clip}}(s_t)-y_t\right)^2
\right)
\right].
$$

When rewards or value targets are normalized, $\epsilon_V=0.2$ is a reasonable
starting point. Otherwise, disable value clipping first because its meaning
depends directly on the reward scale.

### Keep rollout quantities fixed

Compute the old log-probabilities, advantages, and critic targets once before
the optimization epochs. Normalize the advantages across the full rollout
batch, then keep them detached. Recomputing these quantities after every
mini-batch would change the objective while it is being optimized.

### Monitor the update

Track at least the following quantities for every PPO iteration:

- **Approximate KL divergence:** detects an actor update that moved too far.
- **Clip fraction:** the fraction of samples whose probability ratio lies
  outside $[1-\epsilon,1+\epsilon]$.
- **Policy entropy:** detects exploration collapsing too early.
- **Critic loss or explained variance:** detects a critic that is failing to
  fit the return targets.

These signals should be interpreted together. For example, a high clip
fraction with a rapidly increasing KL divergence points to actor updates that
are too aggressive, while a low clip fraction and almost zero KL divergence
may indicate that the batch is being underused.
