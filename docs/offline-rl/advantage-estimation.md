# Advantage Estimation for Weighted Imitation

Advantage-weighted imitation improves on behavior cloning by assigning more
weight to good dataset actions. The central difficulty is that an offline
dataset contains rewards and transitions, not advantage labels.

The [offline RL overview](index.md) introduces distribution shift, OOD actions,
and advantage-weighted imitation. This page continues from those ideas and
explains how AWR, AWAC, and IQL obtain advantage estimates. It also explains
how IDQL combines an IQL critic with a diffusion behavior policy.

## The weighted imitation objective

The advantage of action $a$ under policy $\pi$ is

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

A positive advantage means that $a$ is better than the policy's average action
at the same state. Advantage-weighted BC uses

$$
w(s,a)=\exp\left(\frac{\widehat{A}(s,a)}{\lambda}\right)
$$

and optimizes

$$
\max_\theta
\mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[
w(s,a)\log\pi_\theta(a\mid s)
\right].
$$

The temperature $\lambda>0$ controls how strongly the actor favors actions
with high estimated advantage. Implementations usually clip the weight at
$w_{\max}$ to stop a few samples from dominating an update.

The policy update is simple. The hard question is how to obtain
$\widehat{A}(s,a)$. There are three common answers.

## Monte Carlo advantages: AWR

If the dataset contains complete trajectories, compute the reward-to-go

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

Fit a value baseline to the recorded returns,

$$
\min_\psi
\mathbb{E}_{(s_t,a_t)\sim\mathcal{D}}
\left[(V_\psi(s_t)-G_t)^2\right],
$$

and estimate

$$
\widehat{A}_t=G_t-V_\psi(s_t).
$$

[Advantage-Weighted Regression
(AWR)](https://arxiv.org/abs/1910.00177) uses this estimator.

### AWR walkthrough

**Step 1: Compute returns.** Compute $G_t$ for every transition in each
recorded trajectory.

**Step 2: Fit the baseline.** Regress $V_\psi(s_t)$ onto $G_t$.

**Step 3: Estimate advantages.** Compute
$\widehat{A}_t=G_t-V_\psi(s_t)$.

**Step 4: Compute weights.** Use

$$
w_t=\min\left(
w_{\max},
\exp\left(\frac{\widehat{A}_t}{\lambda}\right)
\right).
$$

**Step 5: Update the actor.** Optimize

$$
\max_\theta
\mathbb{E}_{(s_t,a_t)\sim\mathcal{D}}
\left[
w_t\log\pi_\theta(a_t\mid s_t)
\right].
$$

AWR is simple and never evaluates a critic at OOD actions. Its main weakness
is the return estimator. Monte Carlo returns can be noisy, and the recorded
continuation follows the behavior policy $\pi_\beta$. The resulting advantage
is therefore closer to $A^{\pi_\beta}$ than to the advantage of an improved
policy.

## TD advantages: AWAC

[Advantage Weighted Actor Critic
(AWAC)](https://arxiv.org/abs/2006.09359) uses TD learning to estimate the
Q-function of the current policy $\pi_k$. This can evaluate a policy stronger
than $\pi_\beta$ and stitch together useful transitions from different
trajectories.

### AWAC walkthrough

**Step 1: Sample a transition.** Draw
$(s,a,r,s')\sim\mathcal{D}$.

**Step 2: Update the critic.** Sample
$a'\sim\pi_k(\cdot\mid s')$ and form

$$
y=r+\gamma
\mathbb{E}_{a'\sim\pi_k(\cdot\mid s')}
\left[Q_{\bar\phi}(s',a')\right].
$$

Fit the critic with

$$
\mathcal{L}_Q(\phi)
=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}
\left[(Q_\phi(s,a)-y)^2\right].
$$

**Step 3: Estimate the current-policy baseline.** Compute

$$
\widehat{V}^{\pi_k}(s)
=\mathbb{E}_{\bar a\sim\pi_k(\cdot\mid s)}
\left[Q_\phi(s,\bar a)\right].
$$

**Step 4: Estimate a dataset action's advantage.** Use

$$
\widehat{A}^{\pi_k}(s,a)
=Q_\phi(s,a)-\widehat{V}^{\pi_k}(s),
\qquad (s,a)\sim\mathcal{D}.
$$

**Step 5: Update the actor.** Treat the advantage weights as fixed and optimize

$$
\max_\theta
\mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[
\log\pi_\theta(a\mid s)
\exp\left(\frac{\widehat{A}^{\pi_k}(s,a)}{\lambda}\right)
\right].
$$

After the actor update, set $\pi_k\leftarrow\pi_\theta$ and repeat.

AWAC trains its actor only on dataset actions. However, the critic target and
the value baseline query actions sampled from $\pi_k$. Those actions may be
OOD, so extrapolation error can corrupt both the Bellman target and the actor
weights.

Replacing $a'\sim\pi_k$ with the recorded next action avoids the OOD query, but
then the critic evaluates the behavior policy or behavior-policy mixture. The
key question is whether the learner can represent a policy better than
$\pi_\beta$ without querying Q-values at OOD actions.

## In-sample advantages: IQL

[Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169) answers this
question with an upper-expectile value function. For a fixed state $s$, the
dataset supplies Q-values for actions
$a\sim\pi_\beta(\cdot\mid s)$.

Ordinary squared-error regression gives their mean:

$$
\operatorname*{argmin}_{v}
\mathbb{E}_{a\sim\pi_\beta(\cdot\mid s)}
\left[(Q_\phi(s,a)-v)^2\right]
=\mathbb{E}_{a\sim\pi_\beta(\cdot\mid s)}
\left[Q_\phi(s,a)\right].
$$

If $Q_\phi=Q^{\pi_\beta}$, this mean is $V^{\pi_\beta}(s)$. IQL instead uses
the asymmetric expectile loss

$$
L_2^\tau(u)
=\left|\tau-\mathbb{1}(u<0)\right|u^2,
$$

where $u=Q_\phi(s,a)-V_\psi(s)$. When $\tau>0.5$, positive residuals receive
more weight, so $V_\psi(s)$ moves toward the high-Q part of the dataset-action
distribution. It is a soft estimate of the better actions inside data support,
not an exact maximum.

### General expectile calculation

For scalar samples $x_1,\ldots,x_n$, define

$$
e_\tau
=\operatorname*{argmin}_{v}
\sum_{i=1}^{n}
\left|\tau-\mathbb{1}(x_i-v<0)\right|(x_i-v)^2.
$$

For a candidate $v$, assign

$$
w_i(v)=
\begin{cases}
\tau, & x_i\geq v,\\
1-\tau, & x_i<v.
\end{cases}
$$

The solution satisfies

$$
\sum_{i=1}^{n}w_i(e_\tau)(x_i-e_\tau)=0,
$$

or equivalently

$$
e_\tau=
\frac{
\tau\displaystyle\sum_{x_i\geq e_\tau}x_i
+(1-\tau)\displaystyle\sum_{x_i<e_\tau}x_i
}{
\tau\displaystyle\sum_{x_i\geq e_\tau}1
+(1-\tau)\displaystyle\sum_{x_i<e_\tau}1
}.
$$

The groups depend on $e_\tau$, so a tabular solver repeatedly recomputes the
groups and weighted mean. A neural network minimizes the expectile loss by
gradient descent over minibatches.

!!! note "Expectile walkthrough"

    Suppose the dataset-action Q-values at one state are $[0,0,10]$. Their
    ordinary mean is $10/3\approx3.33$.

    With $\tau=0.8$ and a candidate $v\in[0,10]$, the expectile loss is

    $$
    \begin{aligned}
    \mathcal{L}(v)
    &=0.2(0-v)^2+0.2(0-v)^2+0.8(10-v)^2\\
    &=0.4v^2+0.8(10-v)^2.
    \end{aligned}
    $$

    Setting its derivative to zero gives

    $$
    0.8v+1.6(v-10)=0
    \quad\Longrightarrow\quad
    v=\frac{20}{3}\approx6.67.
    $$

    The upper expectile lies above the mean but below the maximum. The
    resulting advantages are approximately

    $$
    \widehat A(s,a_1)=-6.67,\qquad
    \widehat A(s,a_2)=-6.67,\qquad
    \widehat A(s,a_3)=3.33.
    $$

    Weighted BC therefore strongly favors $a_3$ while evaluating only dataset
    actions.

### IQL walkthrough

Let $Q_\phi$ be the critic, $Q_{\bar\phi}$ its target, and $V_\psi$ the
expectile value function.

**Step 1: Sample a transition.** Draw
$(s,a,r,s')\sim\mathcal{D}$.

**Step 2: Update the value function.** Compute

$$
u=Q_{\bar\phi}(s,a)-V_\psi(s)
$$

and minimize

$$
\mathcal{L}_V(\psi)
=\mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[
\left|\tau-\mathbb{1}(u<0)\right|u^2
\right].
$$

**Step 3: Update the critic.** Use

$$
y=r+\gamma V_\psi(s')
$$

for non-terminal transitions, and $y=r$ for true terminal transitions. Then
minimize

$$
\mathcal{L}_Q(\phi)
=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}
\left[(Q_\phi(s,a)-y)^2\right].
$$

**Step 4: Estimate advantages.** Compute

$$
\widehat A(s,a)=Q_{\bar\phi}(s,a)-V_\psi(s).
$$

**Step 5: Update the actor.** Define

$$
w(s,a)=\min\left(
w_{\max},
\exp\left(\frac{\widehat A(s,a)}{\beta}\right)
\right)
$$

and minimize

$$
\mathcal{L}_\pi(\theta)
=-\mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[w(s,a)\log\pi_\theta(a\mid s)\right].
$$

**Step 6: Update the target critic.** Apply

$$
\bar\phi\leftarrow\rho\bar\phi+(1-\rho)\phi,
$$

where $\rho$ is close to one, and repeat with another minibatch.

IQL avoids OOD Q-queries throughout training:

- the value update uses dataset $(s,a)$ pairs;
- the critic bootstraps from $V_\psi(s')$, not $Q(s',a')$;
- the actor regresses onto dataset actions.

The learned actor may generalize to unseen actions at execution time, but IQL
never needs to evaluate their Q-values during training.

## Diffusion policy extraction: IDQL

[Implicit Diffusion Q-Learning
(IDQL)](https://arxiv.org/abs/2304.10573) is different from IQL. IQL usually
fits an explicit policy with advantage-weighted regression. IDQL instead fits
an expressive diffusion model to the behavior distribution and uses the
critic to select among its action samples.

This separation is useful when the dataset contains a multimodal action
distribution. A simple Gaussian actor may average distinct good actions,
whereas a diffusion model can represent several action modes.

Let $\mu_\omega(a\mid s)$ be the diffusion behavior model. For a clean dataset
action $a$, a diffusion step $k$, and Gaussian noise
$\epsilon\sim\mathcal{N}(0,I)$, construct the noisy action

$$
a_k
=\sqrt{\bar\alpha_k}a
+\sqrt{1-\bar\alpha_k}\epsilon.
$$

The noise predictor $\epsilon_\omega$ is trained with the denoising loss

$$
\mathcal{L}_\mu(\omega)
=\mathbb{E}_{(s,a)\sim\mathcal{D},\,k,\,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\omega(a_k,s,k)
\right\|_2^2
\right].
$$

The schedule $\bar\alpha_k$ controls how much noise is added at diffusion step
$k$. This objective learns the dataset's conditional action distribution; it
does not use Q-values to change the diffusion training loss.

### IDQL walkthrough

**Step 1: Train the value function and critic.** Use the IQL value and critic
updates described above. This produces $V_\psi(s)$ and $Q_{\bar\phi}(s,a)$
without evaluating OOD actions during critic training.

**Step 2: Train the diffusion behavior model.** Minimize
$\mathcal{L}_\mu(\omega)$ on the state-action pairs in $\mathcal{D}$. The model
learns to generate actions that resemble the dataset actions at each state.

**Step 3: Generate candidate actions.** At evaluation time, observe $s$ and
draw $N$ candidates:

$$
a_i\sim\mu_\omega(\cdot\mid s),
\qquad i=1,\ldots,N.
$$

**Step 4: Score the candidates.** For the expectile loss, the implicit-policy
weight is

$$
w_i
=\left|
\tau-\mathbb{1}
\left(Q_{\bar\phi}(s,a_i)<V_\psi(s)\right)
\right|.
$$

Thus, candidates above the learned value receive weight $\tau$, while those
below it receive weight $1-\tau$. Normalize the weights as

$$
p_i=\frac{w_i}{\sum_{j=1}^{N}w_j}.
$$

**Step 5: Select an action.** Sample an index from the categorical
distribution $(p_1,\ldots,p_N)$ and execute its candidate action. A common
deterministic alternative is

$$
a=\operatorname*{argmax}_{a_i}Q_{\bar\phi}(s,a_i).
$$

IDQL may evaluate imperfect diffusion samples at policy-extraction time, so
the diffusion model must fit the behavior distribution well. Poor samples can
be OOD for the critic and receive unreliable Q-values. This is why diffusion
model capacity and regularization matter.

## Compare the methods

- **AWR:** no OOD Q-query, but noisy returns and behavior-policy continuation.
- **AWAC:** estimates the current policy with TD learning, but queries Q-values
  at actions sampled from that policy.
- **IQL:** uses TD learning and an upper expectile while evaluating only
  dataset actions during training.
- **IDQL:** uses the IQL critic, models the behavior policy with a diffusion
  model, and selects among sampled candidate actions during policy extraction.

AWR, AWAC, and IQL differ mainly in the source of the advantage estimate. IDQL
changes how the policy is represented and extracted from an IQL critic.
