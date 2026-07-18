# Off-policy Learning and Importance Sampling

## Why off-policy correction is needed

Let $\pi_\theta$ be the **target policy** that we want to evaluate or improve,
and let $\mu$ be the **behavior policy** that generated the data. On-policy
learning uses $\mu=\pi_\theta$. Off-policy learning allows $\mu\neq\pi_\theta$,
which makes it possible to reuse old trajectories, learn from a replay buffer,
or learn from data collected by another agent.

The difficulty is distribution mismatch. An expectation under the target
policy generally cannot be estimated by simply averaging samples from the
behavior policy:

$$
\mathbb{E}_{x\sim p_\pi}[f(x)]
\neq
\mathbb{E}_{x\sim p_\mu}[f(x)].
$$

Importance sampling corrects this mismatch by assigning each behavior-policy
sample a likelihood ratio.

## Importance sampling

For a discrete random variable,

$$
\begin{aligned}
\mathbb{E}_{x\sim p_\pi}[f(x)]
&=\sum_x p_\pi(x)f(x)\\
&=\sum_x p_\mu(x)
\frac{p_\pi(x)}{p_\mu(x)}f(x)\\
&=\mathbb{E}_{x\sim p_\mu}
\left[w(x)f(x)\right],
\end{aligned}
$$

where

$$
w(x)=\frac{p_\pi(x)}{p_\mu(x)}
$$

is the **importance weight**. The same derivation applies to continuous random
variables, with probability mass functions replaced by probability densities.

Given $N$ independent samples $x_i\sim p_\mu$, the ordinary importance-sampling
estimator is

$$
\widehat{I}_{\mathrm{IS}}
=\frac{1}{N}\sum_{i=1}^{N}w(x_i)f(x_i).
$$

This estimator is unbiased when the required expectations exist and the
behavior distribution covers the target distribution.

!!! warning "Support condition"
    Importance sampling requires $p_\mu(x)>0$ whenever $p_\pi(x)f(x)\neq 0$.
    If the target distribution assigns probability to an event that the
    behavior distribution can never generate, no finite importance weight can
    recover the missing information.

## From action ratios to trajectory ratios

For a trajectory
$\tau=(s_1,a_1,\ldots,s_T,a_T,s_{T+1})$, its probability under policy $\pi$
is

$$
p_\pi(\tau)
=\rho_0(s_1)\prod_{t=1}^{T}
\pi(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

Assume that the target and behavior policies interact with the same environment
and use the same initial-state distribution. In the trajectory likelihood
ratio, the initial-state and transition terms cancel:

$$
\begin{aligned}
w(\tau)
&=\frac{p_{\pi_\theta}(\tau)}{p_\mu(\tau)}\\
&=\prod_{t=1}^{T}
\frac{\pi_\theta(a_t\mid s_t)}{\mu(a_t\mid s_t)}\\
&=\prod_{t=1}^{T}\rho_t,
\end{aligned}
$$

where

$$
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\mu(a_t\mid s_t)}
$$

is the one-step importance ratio. Therefore, with
$R(\tau)=\sum_{t=1}^{T}\gamma^{t-1}r_t$, the target policy's expected return
can be written using behavior-policy trajectories:

$$
J(\theta)
=\mathbb{E}_{\tau\sim p_{\pi_\theta}}[R(\tau)]
=\mathbb{E}_{\tau\sim p_\mu}
\left[
\left(\prod_{t=1}^{T}\rho_t\right)R(\tau)
\right].
$$

This identity is the basic bridge from on-policy expectations to off-policy
data.

## Per-decision importance sampling

A full trajectory ratio contains ratios for actions that occur after an
earlier reward. Those later actions cannot affect that reward, and their ratios
have conditional expectation one. We can therefore weight each reward only by
the ratios up to the time at which that reward is observed:

$$
J(\theta)
=\mathbb{E}_{\tau\sim p_\mu}
\left[
\sum_{t=1}^{T}
\gamma^{t-1}
\left(\prod_{k=1}^{t}\rho_k\right)r_t
\right].
$$

This is **per-decision importance sampling**. It remains unbiased under the
same support condition, while avoiding unnecessary future ratios. However, the
cumulative product can still have high variance, especially for long horizons
or when $\pi_\theta$ and $\mu$ differ substantially.

## Why importance sampling can have high variance

The mean importance weight is one:

$$
\mathbb{E}_{x\sim p_\mu}[w(x)]=1.
$$

This does not mean that the weights are close to one. Most samples may receive
tiny weights while a few rare samples receive extremely large weights. Products
of action ratios amplify this effect over time, so an estimator can be unbiased
but too noisy to be useful with a finite batch.

Two common diagnostics or modifications are:

1. **Effective sample size**

   $$
   N_{\mathrm{eff}}
   =\frac{\left(\sum_{i=1}^{N}w_i\right)^2}
   {\sum_{i=1}^{N}w_i^2}.
   $$

   A small value indicates that only a few samples dominate the estimate.

2. **Self-normalized importance sampling**

   $$
   \widehat{I}_{\mathrm{SNIS}}
   =\frac{\sum_{i=1}^{N}w_i f(x_i)}
   {\sum_{i=1}^{N}w_i}.
   $$

   Self-normalization often reduces variance, but it introduces finite-sample
   bias. It is consistent under standard regularity conditions.

Clipping or truncating large ratios can reduce variance further:

$$
\bar\rho_t=\min(c,\rho_t).
$$

The trade-off is bias: after clipping, the estimator is no longer the exact
importance-sampling identity unless an additional correction is applied.

## Connection to off-policy policy gradients

Let
$G_t=\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}$. The same change of measure can be
applied to the policy-gradient estimator:

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\tau\sim p_\mu}
\left[
\left(\prod_{t=1}^{T}\rho_t\right)
\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t
\right].
$$

This expression is exact under the support condition, but the full trajectory
ratio usually has prohibitive variance. Practical off-policy algorithms avoid
using this estimator directly: they use value functions, shorter-horizon
corrections, clipped ratios, or objectives designed to tolerate data from an
older behavior policy.

The central idea is nevertheless simple:

$$
\boxed{
\text{target-policy expectation}
=\text{behavior-policy expectation}\times\text{likelihood ratio}
}
$$

Importance sampling corrects distribution mismatch; the main practical problem
is controlling the variance introduced by that correction.
