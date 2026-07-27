# Actor-Critic

Actor-critic algorithms pair a policy (the actor) with a value estimate (the critic).


## Revisit Value Function

The policy gradient needs one number per sampled action: a weight saying how
good that action was. Value functions are exactly those numbers, so we review
them before building a critic.

### Three value functions

Fix a policy $\pi_\theta$ and a discount factor $\gamma\in(0,1]$.

The **action-value function** $Q^\pi(s_t,a_t)$ is the return we expect if we
take action $a_t$ in state $s_t$ and then follow $\pi_\theta$ until the episode
ends:

$$
Q^\pi(s_t,a_t)
=\mathbb{E}_{\tau\sim p_\theta}
\left[\sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'},a_{t'})
\;\middle|\; s_t,a_t\right].
$$

The **state-value function** $V^\pi(s_t)$ is the return we expect from state
$s_t$ if we follow $\pi_\theta$ until the episode ends. It is the same sum as
$Q^\pi$, but we no longer fix the action at time $t$:

$$
V^\pi(s_t)
=\mathbb{E}_{\tau\sim p_\theta}
\left[\sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'},a_{t'})
\;\middle|\; s_t\right].
$$

We can also average $Q^\pi$ over actions the policy might take in $s_t$. The
expectation is the same whether actions are discrete or continuous; only the
way we expand it changes:

$$
\begin{aligned}
V^\pi(s_t)
&=\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}
\left[Q^\pi(s_t,a_t)\right]\\
&=\sum_{a}\pi_\theta(a\mid s_t)Q^\pi(s_t,a)
&&\text{(discrete $\mathcal{A}$)}\\
&=\int_{\mathcal{A}}\pi_\theta(a\mid s_t)Q^\pi(s_t,a)\,da
&&\text{(continuous $\mathcal{A}$)}.
\end{aligned}
$$

So $Q^\pi$ scores one specific action in a state, and $V^\pi$ scores the state
itself: how well the policy does from $s_t$ on average.

The **advantage function** is their difference:

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

It answers a simple question: is action $a_t$ better or worse than what the
policy would usually do in this state? 
A positive advantage means better than average, a negative one means worse. 
Because $V^\pi$ is the average of $Q^\pi$ over the policy's all possible actions, 
i.e., for each state $s$, the expectation of the advantage functionis zero: 

$$
\mathbb{E}_{a\sim\pi_\theta}[A^\pi(s,a)]=0.
$$


### From reward-to-go to advantage

In the previous chapter each action was weighted by
$G_{i,t}-b(\mathbf{s}_{i,t})$, where
$G_{i,t}=\sum_{t'=t}^{T}\gamma^{t'-t}r(\mathbf{s}_{i,t'},\mathbf{a}_{i,t'})$ is
the reward-to-go. That reward-to-go is the return we happened to observe in one
trajectory, and its average over many trajectories is exactly $Q^\pi$:

$$
\mathbb{E}\left[G_{i,t}\;\middle|\;\mathbf{s}_{i,t},\mathbf{a}_{i,t}\right]
=Q^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t}).
$$

In other words, the reward-to-go is a one-sample estimate of $Q^\pi$. It is
unbiased, but a single roll-out of a random policy in a random environment can
land far from the average, which is where the variance comes from. If we use
$V^\pi$ as the baseline and replace the sampled return by the value it
estimates, the weight becomes the advantage:

$$
\nabla_\theta J(\theta)
\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})
A^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t}).
$$

All three weights estimate the same gradient. They differ only in how much is
sampled and how much is computed from a value function:

| Weight | Unbiased? | Variance | What we need |
| --- | --- | --- | --- |
| $G_{i,t}$ | yes | highest | samples only |
| $G_{i,t}-V^\pi(\mathbf{s}_{i,t})$ | yes | lower | $V^\pi$ |
| $A^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t})$ | yes if $Q^\pi,V^\pi$ are exact | lowest | $Q^\pi$ and $V^\pi$ |

The catch is that we never know $A^\pi$; we have to learn an approximation of
it. That learned approximation is the critic, and because it is only an
approximation it brings back some bias.

### One value function is enough

We do not need to learn $Q^\pi$ and $V^\pi$ separately. Taking action $a_t$
gives an immediate reward and lands us in a next state, from which the policy
simply continues. That is the Bellman equation:

$$
Q^\pi(s_t,a_t)
=r(s_t,a_t)+\gamma\,\mathbb{E}_{s_{t+1}\sim P(\cdot\mid s_t,a_t)}
\left[V^\pi(s_{t+1})\right].
$$

Substituting it into $A^\pi=Q^\pi-V^\pi$ removes $Q^\pi$ entirely:

$$
A^\pi(s_t,a_t)
=r(s_t,a_t)+\gamma\,\mathbb{E}_{s_{t+1}}
\left[V^\pi(s_{t+1})\right]-V^\pi(s_t).
$$

We already have one sample of that inner expectation: the next state the
environment actually gave us. Using it directly,

$$
\widehat{A}^\pi(s_t,a_t)
=r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)
=\delta_t.
$$

This quantity $\delta_t$ is called the **temporal-difference (TD) error**. It
compares what we thought the state was worth, $V^\pi(s_t)$, with a better
estimate built from the reward we just received. Only $V^\pi$ appears in it, so
one network $V_\phi^\pi(s)\approx V^\pi(s)$ is enough to feed advantages to the
actor. That network is the critic.

### Fitting the critic

Training the critic is ordinary supervised regression: predict a target
$y_{i,t}$ from a state.

$$
\mathcal{L}(\phi)
=\frac{1}{2N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\left(V_\phi^\pi(\mathbf{s}_{i,t})-y_{i,t}\right)^2.
$$

There are two common ways to build the target:

$$
y_{i,t}^{\mathrm{MC}}=G_{i,t},
\qquad
y_{i,t}^{\mathrm{TD}}
=r_{i,t}+\gamma\,V_{\phi^-}^\pi(\mathbf{s}_{i,t+1}).
$$

The Monte Carlo target uses the full observed return. It is unbiased, but it
carries all the variance of the return and needs complete episodes. The TD
target uses one reward plus the critic's own estimate of the next state, an idea
called **bootstrapping**. It has much lower variance and works from single
transitions, but it is biased while the critic is still wrong, because a wrong
prediction is being used to build its own target. If the transition ends the
episode there is no next state, so the bootstrap term is dropped and
$y_{i,t}=r_{i,t}$.

!!! note
    Two things to get right in code. First, the target is a constant:
    $\phi^-$ means `stopgrad`, written `detach()` in PyTorch, so gradients do
    not flow through $V^\pi_{\phi^-}(\mathbf{s}_{i,t+1})$. Second, $V^\pi$ is
    defined for the *current* policy. Every actor update changes what the
    critic should predict, which is why on-policy actor-critic methods refit
    the critic on freshly collected data instead of trusting an old value
    estimate.

The TD error $\delta_t$ looks only one step ahead, so it has the least variance
and the most bias. The Monte Carlo return looks all the way to the end of the
episode, so it has the most variance and no bias. Everything in between is
covered by $n$-step returns and generalized advantage estimation.


## A2C / A3C

A3C introduced asynchronous workers; A2C is its synchronized, vectorized counterpart.

## Soft Actor-Critic

SAC is an off-policy actor-critic method that optimizes reward and policy entropy.

### Notes to add

- $n$-step returns and generalized advantage estimation
- Twin Q-functions and temperature tuning
