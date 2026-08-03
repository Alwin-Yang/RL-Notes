# Actor-Critic

Actor-critic algorithms pair a policy (the actor) with a value estimate (the critic).


## Three value functions

The policy gradient needs one number per sampled action: a weight saying how
good that action was. Value functions are exactly those numbers, so we review
them before building a critic.

### Basic concept

- Fix a policy $\pi_\theta$ and a discount factor $\gamma\in(0,1]$.

- Expectations over $\tau\sim p_\theta$ mean: from the current step onward, at
each $t'$ sample $a_{t'}\sim\pi_\theta(\cdot\mid s_{t'})$ and then
$s_{t'+1}\sim P(\cdot\mid s_{t'},a_{t'})$ until the episode ends. The policy
chooses actions only; next states come from the environment. Writing
$(s_{t'},a_{t'})\sim\pi_\theta$ would wrongly treat $\pi_\theta$ as a joint
law over states and actions. Instead, $\mathbb{E}_{\tau\sim p_\theta}[\cdot]$
is shorthand for that rollout. It is the same as sampling actions and transitions
step by step:

$$
\mathbb{E}_{\tau\sim p_\theta}[X]
=\mathbb{E}_{\substack{a_{t'}\sim\pi_\theta(\cdot\mid s_{t'})\\ s_{t'+1}\sim P(\cdot\mid s_{t'},a_{t'})}}[X],
$$

!!! note "$\gamma$ changes the MDP, not just the arithmetic"
    Discounting is usually introduced as a way to keep an infinite sum finite,
    but it also redefines the task. Using $\gamma$ is equivalent to solving an
    undiscounted MDP in which every step has probability $1-\gamma$ of ending
    the episode. Surviving to step $t$ then has probability $\gamma^{t}$, which
    is exactly the weight that step's reward receives:

    $$
    \mathbb{E}\left[\sum_{t=0}^{K}r_t\right]
    =\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^{t}r_t\right],
    \qquad K\sim\mathrm{Geom}(1-\gamma).
    $$

    Every power of $\gamma$ is a survival probability. The mean lifetime is
    $1/(1-\gamma)$, which is where the effective horizon comes from: about 100
    steps for $\gamma=0.99$, though the geometric distribution is skewed, so the
    median is only around 69. A smaller $\gamma$ therefore does more than reduce
    variance. It makes the optimal policy more myopic, unwilling to pay a
    near-term cost for a distant reward, and the policy we end up with is optimal
    for the discounted objective rather than for the total reward we may actually
    care about.

### Action-value function

$Q^\pi(s_t,a_t)$ is the return we expect if we take action $a_t$ in state $s_t$
and then follow $\pi_\theta$ until the episode ends:

$$
Q^\pi(s_t,a_t)
=\mathbb{E}_{\tau\sim p_\theta}
\left[\sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'},a_{t'})
\;\middle|\; s_t,a_t\right].
$$

### State-value function

$V^\pi(s_t)$ is the return we expect from state $s_t$ if we follow $\pi_\theta$
until the episode ends. It is the same sum as $Q^\pi$, but we no longer fix the
action at time $t$:

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

### Advantage function

$A^\pi(s_t,a_t)$ is the difference between the two:

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

It answers a simple question: is action $a_t$ better or worse than what the
policy would usually do in this state? 
A positive advantage means better than average, a negative one means worse. 
Because $V^\pi$ is the average of $Q^\pi$ over the policy's all possible actions, 
i.e., for each state $s$, the expectation of the advantage function is zero: 

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}[A^\pi(s,a)]=0.
$$


## From reward-to-go to advantage

[Policy Optimization](index.md) defines what we are
optimizing: the expected return of trajectories drawn from $\pi_\theta$,

$$
J(\theta)
=\mathbb{E}_{\tau\sim p_\theta}
\left[\sum_{t=1}^{T}r(s_t,a_t)\right].
$$

Actor-critic methods still perform gradient ascent on this same $J(\theta)$. The
only change is how we estimate $\nabla_\theta J(\theta)$: we replace raw
returns with value-function baselines or advantages to cut variance.

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
land far from the average, which is where the variance comes from.

The weight therefore improves in three steps. The first is the estimator from
the previous chapter, which uses the sampled return:

$$
\nabla_\theta J(\theta)
\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})\,
G_{i,t}.
$$

The second replaces that single sample by the mean it estimates, $Q^\pi$. Note
the direction: $G_{i,t}$ is the noisy sample and $Q^\pi$ is its exact mean, so
swapping them leaves the expected gradient unchanged while the noise of the
roll-out after time $t$ is gone:

$$
\nabla_\theta J(\theta)
\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})\,
Q^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t}).
$$

The third subtracts the state-dependent baseline $V^\pi$. As in the previous
chapter a baseline leaves the expected gradient unchanged, and this particular
one centers the weight on how the policy usually does in that state, which turns
$Q^\pi$ into the advantage:

$$
A^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t})
=Q^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t})-V^\pi(\mathbf{s}_{i,t}),
$$

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
| $Q^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t})$ | yes if $Q^\pi$ is exact | lower | $Q^\pi$ |
| $A^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t})$ | yes if $Q^\pi,V^\pi$ are exact | lowest | $Q^\pi$ and $V^\pi$ |

The catch is that we never know $A^\pi$; we have to learn an approximation of
it. That learned approximation is the critic, and because it is only an
approximation it brings back some bias.

## One value function is enough

We do not need to learn $Q^\pi$ and $V^\pi$ separately. Two relations connect
them, and we want the one that leaves only $V^\pi$. The relation
$V^\pi(s_t)=\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}[Q^\pi(s_t,a_t)]$
points the wrong way: it expresses $V^\pi$ through $Q^\pi$, so the function we
would still have to learn is $Q^\pi$. Learning $Q^\pi$ costs us three things.

First, $Q^\pi$ is defined on $\mathcal{S}\times\mathcal{A}$ while $V^\pi$ is
defined on $\mathcal{S}$ alone, so the same amount of data has to cover a much
larger input space.

Second, the advantage still needs $V^\pi$, and recovering it from $Q^\pi$ means
averaging over the action space. That average is an exact sum when the actions
are discrete, but it requires sampling or integration when they are continuous,
which costs extra computation and adds noise of its own.

Third, a TD target for $Q^\pi$ needs a next *action* as well as a next state,
$r_t+\gamma Q^\pi(s_{t+1},a_{t+1})$ with $a_{t+1}\sim\pi_\theta$, and that extra
sampled action brings extra variance. A target for $V^\pi$ needs only the next
state, which the environment already handed us.

None of this makes $Q^\pi$ the wrong object to learn. It is the right one when
the critic has to work with off-policy data or to produce the policy itself,
which is what [Q-learning](../q-learning/index.md) and
[SAC](../actor-critic/sac.md) do;
the [Q&A](actor-critic-q-and-a.md) covers that trade-off. But for on-policy
advantage
estimation $V^\pi$ is all we need, and the other direction of the relation takes
us there. Taking action $a_t$ gives an immediate reward and lands us in a next
state, from which the policy simply continues. That is the Bellman equation:

$$
Q^\pi(s_t,a_t)
=r(s_t,a_t)+\gamma\,\mathbb{E}_{s_{t+1}\sim P(\cdot\mid s_t,a_t)}
\left[V^\pi(s_{t+1})\right].
$$

Substituting it into $A^\pi=Q^\pi-V^\pi$ removes $Q^\pi$ entirely:

$$
A^\pi(s_t,a_t)
=r(s_t,a_t)+\gamma\,\mathbb{E}_{s_{t+1}\sim P(\cdot\mid s_t,a_t)}
\left[V^\pi(s_{t+1})\right]-V^\pi(s_t).
$$

We already have one sample of that inner expectation: the next state the
environment actually gave us. Writing $r_t=r(s_t,a_t)$ for the reward observed
on that transition and using the sampled next state directly,

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

!!! note "Notation: $r(s_t,a_t)$ versus $r_t$"
    $r(s,a)$ is a function. Given a state and an action it returns a fixed
    number, and if the reward itself is random it returns the mean,
    $r(s,a)=\mathbb{E}[R\mid s,a]$. In contrast $r_t$ is the reward we actually
    observed at step $t$, one draw of that random variable, so
    $\mathbb{E}[r_t\mid s_t,a_t]=r(s_t,a_t)$.

    This is the same relation as $\mathbb{E}[G_{i,t}\mid\mathbf{s}_{i,t},
    \mathbf{a}_{i,t}]=Q^\pi(\mathbf{s}_{i,t},\mathbf{a}_{i,t})$ and
    $\mathbb{E}[\delta_t\mid s_t,a_t]=A^\pi(s_t,a_t)$, one level down. The rule
    across these notes: a quantity written as a function of state and action is
    exact and has no variance, while a quantity carrying a time index or a hat
    is sampled. Definitions and Bellman equations therefore use $r(s_t,a_t)$,
    and estimators built from a single transition use $r_t$.

## Fitting the critic

Training the critic is ordinary supervised regression: predict a target
$y_{i,t}$ from a state.

$$
\mathcal{L}(\phi)
=\frac{1}{2N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\left(V_\phi^\pi(\mathbf{s}_{i,t})-y_{i,t}\right)^2.
$$

Only the target is left to choose. There are **two** common ways to build it:
add up the rewards we actually observed, or use one observed reward plus the
critic's own prediction.

!!! note "In code"
    The target is a constant, whichever form it takes: it is wrapped in
    `stopgrad`, written `detach()` in PyTorch, so the regression never
    differentiates through it. And $V^\pi$ is defined for the *current* policy,
    so every actor update changes what the critic should predict; on-policy
    methods refit the critic on fresh data instead of trusting an old value
    estimate.

### Monte Carlo target

The Monte Carlo target is the reward-to-go, the discounted sum of the rewards
that actually followed step $t$:

$$
y_{i,t}^{\mathrm{MC}}
=G_{i,t}
=\sum_{t'=t}^{T}\gamma^{t'-t}r_{i,t'}.
$$

With $\gamma=1$ it is simply those rewards added up,
$y_{i,t}^{\mathrm{MC}}=r_{i,t}+r_{i,t+1}+\cdots+r_{i,T}$. Rewards collected
before step $t$ are excluded, since $V^\pi(s_t)$ asks only what we expect from
$s_t$ onward. The whole-trajectory return $\sum_{t'=1}^{T}r_{i,t'}$ is therefore
not a valid target; that is the weight
[REINFORCE](policy-improvement.md) applies to every step.

How can a single sampled return estimate an expectation? Taken literally, the
definition of $V^\pi$ asks us to reset the environment to a state $s$, run $m$
independent rollouts from there, and average their returns:

$$
\widehat{V}^\pi(s)
=\frac{1}{m}\sum_{k=1}^{m}G^{(k)}(s),
\qquad
G^{(k)}(s)=\sum_{j=0}^{H_k}\gamma^{j}r_j^{(k)},
$$

where rollout $k$ starts at $s_0^{(k)}=s$, runs for its own length $H_k$, and
collects $r_j^{(k)}$ at $j$ steps after the reset. Its variance shrinks like
$1/m$, but most environments cannot be reset to an arbitrary state, and each
target would cost $m$ times the interaction. TRPO's vine estimator is one of
the few methods that does exploit a resettable simulator.

In practice $m=1$: one trajectory gives one target per visited step, obtained by
walking backwards through it,

$$
G_{i,T}=r_{i,T},
\qquad
G_{i,t}=r_{i,t}+\gamma G_{i,t+1},
$$

which turns an episode of length $T$ into $T$ training pairs
$(\mathbf{s}_{i,t},G_{i,t})$.

This target is unbiased, but it carries the full variance of the return and it
needs complete episodes, since the sum only closes once the episode ends.

!!! note "Why one target per state is enough"
    The averaging still happens, but across states instead of within one state.
    Least squares has the conditional mean as its minimizer,
    $\arg\min_f\mathbb{E}[(f(s)-y)^2]=\mathbb{E}[y\mid s]$, and
    $\mathbb{E}[G_{i,t}\mid\mathbf{s}_{i,t}]=V^\pi(\mathbf{s}_{i,t})$, so the
    optimum of this regression is exactly $V^\pi$. Shared parameters let similar
    states cancel each other's noise, at the cost of some approximation bias. A
    tabular critic shares nothing, which is why tabular Monte Carlo evaluation
    does need many visits to the same state.

### TD target

The TD target replaces everything after the first reward by the critic's own
estimate of the next state:

$$
y_{i,t}^{\mathrm{TD}}
=r_{i,t}+\gamma\,V_{\phi^-}^\pi(\mathbf{s}_{i,t+1}).
$$

Using a prediction to build the target of that same prediction is called
**bootstrapping**. Here $\phi^-$ marks the target as detached, as the note above
describes. If the transition ends the episode there is no next state, so the
bootstrap term is dropped and $y_{i,t}=r_{i,t}$.

This target needs only a single transition, not a finished episode, and its
variance is much lower because only one sampled reward enters it. The cost is
bias: while the critic is still wrong, a wrong prediction is being used to build
its own target.

## n-step returns and GAE

$\delta_t$ looks one step ahead: least variance, most bias. The MC return looks
all the way to the end of the episode: most variance, no bias. They are the two
ends of one family, and the knob between them is how many real rewards we use
before handing over to the critic.

### n-step targets

Using $n$ observed rewards and then bootstrapping gives the $n$-step target

$$
y_{i,t}^{(n)}
=\sum_{j=0}^{n-1}\gamma^{j}r_{i,t+j}
+\gamma^{n}V_{\phi^-}^\pi(\mathbf{s}_{i,t+n}).
$$

Setting $n=1$ returns the TD target of the previous section, and letting $n$ run
to the end of the episode drops the bootstrap term and leaves the MC target. The
larger $n$ is, the more sampled rewards enter the target: less bias, more
variance.

The matching advantage estimate collapses into a sum of one-step TD errors,

$$
\widehat{A}_t^{(n)}
=\sum_{j=0}^{n-1}\gamma^{j}r_{t+j}+\gamma^{n}V^\pi(s_{t+n})-V^\pi(s_t)
=\sum_{j=0}^{n-1}\gamma^{j}\delta_{t+j},
$$

because each intermediate value $V^\pi(s_{t+1}),\dots,V^\pi(s_{t+n-1})$ appears
once with a plus and once with a minus sign and cancels.

Choosing $n$ means trading the critic's bias against the environment's variance,
so the question is simply how much we trust the critic relative to how noisy the
returns are.

| Situation | Prefer |
| --- | --- |
| Critic already fits well | small $n$ |
| Critic untrained or a poor fit | large $n$ |
| Dense, informative rewards | small $n$ |
| Sparse or delayed rewards | large $n$ |
| Noisy dynamics or a high-entropy policy | small $n$ |
| Nearly deterministic dynamics | large $n$ |
| Long or non-terminating episodes | small $n$ |
| Short episodes, or large batches to average over | large $n$ |

The variance rows follow from where the noise comes from. Every extra sampled
reward in the target brings the randomness of one more action and one more
transition, so the noisier the trajectories, the more a long sum costs us. If the
dynamics are close to deterministic, that sum costs almost nothing and we may as
well take the unbiased end.

The sparse-reward row is the one that bites hardest in practice. With $n=1$ a
reward received at the end of an episode moves backwards only one step per round
of critic fitting, so it can take many updates before early states know anything
about it. A large $n$ carries that reward back many steps at once.

Common practice is not to pick $n$ at all, but to pick $\lambda$ below, whose
effective horizon is roughly $1/(1-\lambda)$. Where $n$ is used directly, as in
A2C and A3C, small values such as 5 are typical.

### Generalized advantage estimation

Instead of committing to one $n$, GAE averages all of them with exponentially
decaying weights:

$$
\widehat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{j=0}^{\infty}(\gamma\lambda)^{j}\delta_{t+j}.
$$

Here $\lambda=0$ recovers $\delta_t$ and $\lambda=1$ recovers the Monte Carlo
advantage, so $\lambda$ is a continuous version of the same knob and is easier to
tune than an integer $n$. Common values are $0.95$ to $0.99$. It costs nothing
extra to compute, since the same backward pass that builds the reward-to-go also
builds this sum:

$$
\widehat{A}_t=\delta_t+\gamma\lambda\widehat{A}_{t+1},
$$

started from $\widehat{A}_T=\delta_T$ and reset to zero at episode boundaries.

!!! warning "Valid on-policy only"
    An $n$-step target uses the $n$ actions that were actually taken after step
    $t$, so those actions have to come from the current policy. Reusing data
    from an older policy needs an importance-sampling correction, which is what
    Retrace and V-trace provide. See
    [off-policy learning](off-policy.md).


## A full algorithm walkthrough

Everything above assembles into one loop over two networks trained side by side:
the actor $\pi_\theta(a\mid s)$ and the critic $V_\phi^\pi(s)$.

1. **Collect data.** Run $\pi_\theta$ in the environment and store a batch of
   trajectories
   $\{(\mathbf{s}_{i,1},\mathbf{a}_{i,1},\ldots,\mathbf{s}_{i,T},\mathbf{a}_{i,T})\}_{i=1}^{N}$.
2. **Fit the critic.** Build a target $y_{i,t}$ for every visited step, then
   take a few gradient steps on
   $\mathcal{L}(\phi)=\frac{1}{2N}\sum_{i,t}(V_\phi^\pi(\mathbf{s}_{i,t})-y_{i,t})^2$.
3. **Evaluate advantages.** For all $i$ and $t$, compute
   $\widehat{A}_{i,t}=r_{i,t}+\gamma V_\phi^\pi(\mathbf{s}_{i,t+1})-V_\phi^\pi(\mathbf{s}_{i,t})$,
   or its $n$-step or GAE version.
4. **Estimate the policy gradient.**
   $\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i,t}\nabla_\theta\log\pi_\theta(\mathbf{a}_{i,t}\mid\mathbf{s}_{i,t})\,\widehat{A}_{i,t}$.
5. **Update the actor.** $\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)$, then
   throw the batch away and go back to step 1.

Steps 2 and 3 are where the choices from this page enter: the target in step 2 is
MC, TD, or $n$-step, and the advantage in step 3 inherits whatever horizon we
picked. Both $y_{i,t}$ and $\widehat{A}_{i,t}$ are detached, so the critic loss
never updates the actor and the actor loss never updates the critic.

Step 5 throws the batch away because the batch is on-policy. Once $\theta$ moves,
$V_\phi^\pi$ no longer describes the current policy and the sampled actions no
longer come from it, so the same data cannot be reused. Off-policy actor-critic
methods differ exactly here: they keep a replay buffer and pay for it with
importance corrections or a $Q$ critic.

Two habits worth adopting from the start: normalize the advantages within a batch
before step 4, which stabilizes the actor's step size, and take several critic
epochs per actor update, since a stale critic biases every advantage in step 3.

## Algorithms

- [A2C / A3C](a2c-a3c.md): the synchronous and asynchronous on-policy versions
  of everything above.
- [Trust Region Policy Optimization](trpo.md): constrains the policy update
  using the KL divergence.
- [Proximal Policy Optimization](ppo.md): approximates a trust-region update
  with a clipped surrogate objective.
- [Soft Actor-Critic](../actor-critic/sac.md): an off-policy actor-critic that
  learns $Q$ and adds an entropy bonus.

## Questions

See the [Actor-Critic Q&A](actor-critic-q-and-a.md) for conceptual questions
about what the critic is for, why it learns $V^\pi$ rather than $Q^\pi$, and
how that differs from Q-learning.
