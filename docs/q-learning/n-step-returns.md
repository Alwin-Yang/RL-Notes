# N-Step Returns

[DQN](dqn.md) uses one observed reward and then bootstraps from its target
network. An $n$-step return uses several observed rewards before it
bootstraps. This can move delayed rewards through the Q-function faster, but it
also changes the target's variance and its relationship to the behavior
policy.

## From a one-step target to an n-step target

For a non-terminal transition, the one-step DQN target is

$$
y_t^{(1)}
=r_t
+\gamma\max_a\widehat{Q}_{\bar\phi}(s_{t+1},a).
$$

Instead of trusting the target critic after one step, we can keep following the
sampled trajectory for $n$ rewards and bootstrap at $s_{t+n}$:

$$
y_t^{(n)}
=\sum_{j=0}^{n-1}\gamma^j r_{t+j}
+\gamma^n
\max_a\widehat{Q}_{\bar\phi}(s_{t+n},a).
$$

The first term is the sampled $n$-step reward

$$
R_t^{(n)}
=\sum_{j=0}^{n-1}\gamma^j r_{t+j}.
$$

The second term estimates everything after those $n$ rewards. Setting $n=1$
recovers the DQN target. If $n$ reaches the end of the episode, there is no
state from which to bootstrap, and the target contains rewards only.

Suppose the episode terminates after $m\leq n$ transitions. The target is then

$$
y_t^{(n)}
=\sum_{j=0}^{m-1}\gamma^j r_{t+j}.
$$

The terminal reward is included, but no value is added after the terminal
state.

## Why use more than one step?

A one-step target moves information backward one transition at a time. If a
reward appears five steps after an action, several rounds of one-step Bellman
updates may be needed before that action receives the information. A five-step
target includes the reward immediately.

Increasing $n$ changes what the target trusts:

- A small $n$ uses fewer sampled rewards and relies more on the critic. It
  usually has lower sampling variance, but more bias when the critic is wrong.
- A large $n$ uses more of the observed trajectory and relies less on the
  critic. It reduces bootstrap bias, but accumulates more randomness from
  rewards, transitions, and behavior actions.
- Taking $n$ to the end of the episode gives a Monte Carlo return. It has no
  bootstrap error, but it must wait for the episode to finish and can have high
  variance.

The useful value of $n$ depends on the task. Larger values can help with sparse
or delayed rewards. Smaller values are often safer when dynamics and rewards
are noisy or when replay data comes from policies that differ substantially
from the current greedy policy.

## The off-policy complication

One-step Q-learning can learn from any observed transition. The stored action
$a_t$ may come from a behavior policy $\mu$, but the target starts acting
greedily immediately after the observed next state $s_{t+1}$.

An $n$-step sample is different. Its rewards and states depend on the actual
intermediate actions

$$
a_{t+1},\ldots,a_{t+n-1},
$$

and those actions were selected by $\mu$. If $\mu$ takes exploratory actions,
the sampled continuation is not the continuation that the greedy target policy
would have produced. A final $\max_a$ at $s_{t+n}$ does not correct this
mismatch.

Exact off-policy $n$-step learning therefore needs a correction for the
intermediate actions. Importance sampling can provide one, but products of
probability ratios can have high variance. The problem is especially severe
when the target policy is deterministic and greedy: a non-greedy behavior
action has zero probability under the target policy.

Methods such as Tree Backup, Retrace, and V-trace replace or truncate these
ratios to obtain more practical multi-step targets. See
[off-policy learning](../policy-optimization/off-policy.md) for the underlying
change-of-measure problem.

!!! warning "Uncorrected n-step replay is approximate"
    Some deep Q-learning implementations use short uncorrected $n$-step
    returns from an $\epsilon$-greedy replay buffer. This often works in
    practice when $n$ is small and the behavior policy is close to greedy, but
    the target is then biased by the intermediate behavior actions.

## Store n-step transitions in replay

A replay buffer must preserve enough temporal information to construct the
target. There are two common designs:

1. Store ordinary one-step transitions together with episode boundaries, then
   sample contiguous sequences of length $n$.
2. Maintain a short queue while interacting with the environment, aggregate
   each sequence into an n-step transition, and store the aggregate in replay.

For the second design, let $\ell_i\leq n$ be the number of rewards available in
sample $i$. It is smaller than $n$ only when the episode terminates early. Store

$$
\left(
\mathbf{s}_i,
\mathbf{a}_i,
\mathbf{R}_i^{(\ell_i)},
\mathbf{s}'_i,
\mathbf{d}_i,
\ell_i
\right),
$$

where

$$
\mathbf{R}_i^{(\ell_i)}
=\sum_{j=0}^{\ell_i-1}\gamma^j\mathbf{r}_{i,j}.
$$

Here, $\mathbf{s}'_i$ is the state reached after $\ell_i$ transitions and
$\mathbf{d}_i$ indicates whether that state is terminal. The DQN target becomes

$$
y_i^{(n)}
=\mathbf{R}_i^{(\ell_i)}
+\gamma^{\ell_i}(1-\mathbf{d}_i)
\max_a\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a).
$$

The exponent must be $\ell_i$, not always $n$, because an early terminal state
shortens the sampled return.

## An n-step DQN walkthrough

The DQN loop changes only how transitions and targets are built:

1. **Collect a short sequence.** Act with the $\epsilon$-greedy behavior policy
   and keep the latest transitions in a queue.
2. **Build an n-step transition.** Once the queue contains $n$ rewards, compute
   their discounted sum and store the aggregate transition in replay. At an
   episode boundary, flush the remaining shorter sequences without
   bootstrapping past the terminal state.
3. **Sample a batch.** Draw aggregated transitions from the replay buffer.
4. **Build n-step targets.** Compute

    $$
    y_i^{(n)}
    =\mathbf{R}_i^{(\ell_i)}
    +\gamma^{\ell_i}(1-\mathbf{d}_i)
    \max_a\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a).
    $$

5. **Update the online critic.** Fit
   $\widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)$ to $y_i^{(n)}$ with the same
   squared-error loss used by DQN.
6. **Update the target network and repeat.** Keep the replay, exploration, and
   target-network schedules unchanged.

For [Double DQN](double-dqn.md), the online network selects the bootstrap action
and the target network evaluates it:

$$
\begin{aligned}
a_i^*
&=\arg\max_a\widehat{Q}_\phi(\mathbf{s}'_i,a),\\
y_i^{(n,\mathrm{Double})}
&=\mathbf{R}_i^{(\ell_i)}
+\gamma^{\ell_i}(1-\mathbf{d}_i)
\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a_i^*).
\end{aligned}
$$

## Questions

See the [Q-Learning Q&A](q-and-a.md) for conceptual questions about off-policy
learning and bootstrap targets.
