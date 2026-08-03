# Trust Region Policy Optimization

Trust region policy optimization (TRPO) is an on-policy actor-critic algorithm.
It maximizes a surrogate objective while explicitly limiting how much the
policy may change in one update.

## Start from the surrogate objective

Set $\theta_{\mathrm{old}}\leftarrow\theta$, collect trajectories with
$\pi_{\theta_{\mathrm{old}}}$, fit the critic, and compute fixed advantage
estimates $\widehat{A}_t$. As in
[PPO](ppo.md#the-surrogate-objective), define

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

and the surrogate objective

$$
\widetilde{J}(\theta)
=
\mathbb{E}_t
\left[
\rho_t(\theta)\widehat{A}_t
\right].
$$

The expectation is estimated from the batch collected by the old policy. The
actor maximizes $\widetilde{J}(\theta)$ while keeping the old policy and
advantage estimates fixed.

## Why the surrogate is only locally reliable

The performance-difference identity compares the new and old policies:

$$
J(\theta)-J(\theta_{\mathrm{old}})
=
\frac{1}{1-\gamma}
\mathbb{E}_{\substack{
s\sim d_\theta\\
a\sim\pi_\theta(\cdot\mid s)
}}
\left[
A^{\pi_{\theta_{\mathrm{old}}}}(s,a)
\right],
$$

where $d_\theta$ is the normalized discounted state-visitation distribution of
the new policy. The exact expression depends on states visited by the new
policy, but the available batch contains states visited by the old policy.

The surrogate objective replaces $d_\theta$ with
$d_{\theta_{\mathrm{old}}}$ and corrects only the action probabilities through
$\rho_t(\theta)$. At $\theta=\theta_{\mathrm{old}}$, the surrogate has the
correct value and policy gradient. After a large update, however, the new
policy may visit different states, so the surrogate can predict an improvement
that does not occur in the environment.

TRPO therefore maximizes the surrogate only inside a small region around the
old policy.

## The trust-region constraint

Define the average KL divergence over states in the current batch:

$$
\overline{D}_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)
=
\mathbb{E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\middle\|
\pi_\theta(\cdot\mid s_t)
\right)
\right].
$$

TRPO solves the constrained problem

$$
\begin{aligned}
\max_\theta\quad
&\widetilde{J}(\theta)
\\
\text{subject to}\quad
&\overline{D}_{\mathrm{KL}}
(\theta_{\mathrm{old}},\theta)
\leq\delta,
\end{aligned}
$$

where $\delta>0$ is the trust-region size. The KL divergence controls the
whole action distribution at each sampled state, rather than only the
probability of the sampled action.

!!! note "Theory and implementation"
    The monotonic-improvement bound behind TRPO uses a worst-case KL divergence
    over states. The practical algorithm uses an average sampled KL divergence
    and local approximations. The update is therefore designed to improve
    stability, but it is not an absolute guarantee that every sampled update
    improves the true return.

## Approximate the constrained problem

The neural-network objective and constraint cannot be solved exactly. Let

$$
x=\theta-\theta_{\mathrm{old}}
$$

be the proposed parameter change. First, approximate the surrogate objective
to first order around $\theta_{\mathrm{old}}$:

$$
\widetilde{J}(\theta_{\mathrm{old}}+x)
\approx
\widetilde{J}(\theta_{\mathrm{old}})
+g^\top x,
$$

where

$$
g
=
\left.
\nabla_\theta\widetilde{J}(\theta)
\right|_{\theta=\theta_{\mathrm{old}}}
$$

is the surrogate policy gradient.

The KL divergence is zero and has zero first derivative when the two policies
are identical. Its first useful approximation is therefore second order:

$$
\overline{D}_{\mathrm{KL}}
(\theta_{\mathrm{old}},\theta_{\mathrm{old}}+x)
\approx
\frac{1}{2}x^\top Hx,
$$

where

$$
H
=
\left.
\nabla_\theta^2
\overline{D}_{\mathrm{KL}}
(\theta_{\mathrm{old}},\theta)
\right|_{\theta=\theta_{\mathrm{old}}}
$$

is the KL Hessian, which corresponds to the Fisher information matrix under
standard policy parameterizations.

Dropping the constant
$\widetilde{J}(\theta_{\mathrm{old}})$ gives the local trust-region problem

$$
\begin{aligned}
\max_x\quad
&g^\top x
\\
\text{subject to}\quad
&\frac{1}{2}x^\top Hx\leq\delta.
\end{aligned}
$$

## Solve for the trust-region step

Introduce a Lagrange multiplier $\mu>0$:

$$
\mathcal{F}(x,\mu)
=
g^\top x
-
\mu\left(\frac{1}{2}x^\top Hx-\delta\right).
$$

At a stationary point,

$$
\nabla_x\mathcal{F}(x,\mu)
=
g-\mu Hx
=0,
$$

so

$$
x
=
\frac{1}{\mu}H^{-1}g.
$$

The solution therefore points in the natural-gradient direction $H^{-1}g$.
Substituting it into the active constraint
$\frac{1}{2}x^\top Hx=\delta$ gives

$$
x^*
=
\sqrt{
\frac{2\delta}
{g^\top H^{-1}g}
}
H^{-1}g.
$$

An ordinary gradient uses Euclidean distance in parameter space. The natural
gradient instead accounts for how a parameter change affects the policy
distribution. This matters because the same-sized parameter step can produce
very different probability changes in different parts of a neural network.

## Compute the step without forming the Hessian

Explicitly constructing and inverting $H$ is too expensive for a neural
network. TRPO uses conjugate gradient to approximately solve

$$
Hv=g
$$

for the direction $v=H^{-1}g$. Conjugate gradient only needs products of the
form $Hu$ for an arbitrary vector $u$. Automatic differentiation can compute
these Fisher-vector products without materializing the full matrix.

Implementations often solve with a damped matrix $H+cI$, where $c>0$ is a small
damping coefficient. Damping improves numerical stability when the sampled
curvature matrix is poorly conditioned.

## Check the step with backtracking line search

The step $x^*$ comes from local approximations and finite-sample estimates.
TRPO therefore tests the actual surrogate objective and sampled KL divergence
before accepting it.

Starting with the full step, test candidates

$$
\theta_{\mathrm{candidate}}
=
\theta_{\mathrm{old}}+\eta^j x^*,
\qquad j=0,1,2,\ldots,
$$

where $\eta\in(0,1)$ is the backtracking coefficient. Accept the first
candidate that

- improves the measured surrogate objective, and
- satisfies
  $\overline{D}_{\mathrm{KL}}
  (\theta_{\mathrm{old}},\theta_{\mathrm{candidate}})\leq\delta$.

If neither condition holds, shrink the step again. If no candidate passes,
keep the old actor parameters.

## A full algorithm walkthrough

TRPO trains an actor $\pi_\theta(a\mid s)$ and a critic $V_\phi^\pi(s)$ in the
following loop:

1. **Collect data.** Set $\theta_{\mathrm{old}}\leftarrow\theta$, run
   $\pi_{\theta_{\mathrm{old}}}$ in the environment, and collect a fresh batch
   of trajectories.
2. **Fit the critic.** Build value targets and take several gradient steps on
   the critic loss.
3. **Estimate advantages.** Compute fixed advantage estimates, usually with
   GAE.
4. **Build the local problem.** Evaluate the surrogate gradient $g$ and define
   the average KL constraint around $\theta_{\mathrm{old}}$.
5. **Find the natural-gradient direction.** Use conjugate gradient and
   Fisher-vector products to approximately solve $Hv=g$.
6. **Scale and check the step.** Scale the direction to satisfy the quadratic
   KL constraint, then use backtracking line search to check the actual
   surrogate improvement and sampled KL divergence.
7. **Update and repeat.** Accept the first valid actor update, discard the
   batch, and collect new data. If line search fails, keep the old actor and
   still collect a new batch.

## Relationship to PPO

TRPO and [PPO](ppo.md) begin with the same on-policy actor-critic data and the
same importance-weighted surrogate objective. They control the actor update in
different ways:

- TRPO imposes an explicit average KL constraint and uses second-order
  information, conjugate gradient, and line search.
- PPO uses a clipped first-order objective that can be optimized with ordinary
  mini-batch gradient steps.

TRPO follows the trust-region constraint more directly. PPO is simpler to
implement because it avoids a second-order solver and line search.
