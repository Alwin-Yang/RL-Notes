# Deep Q-Networks

A deep Q-network (DQN) implements the critic from
[Q-learning](index.md) with a neural network. Its parameters are $\phi$, and
its prediction $\widehat{Q}_\phi(s,a)$ estimates the value of taking action
$a$ in state $s$ and then acting greedily.

## Replace the Q-table with a neural network

In tabular Q-learning, every state-action pair has a separate entry. Updating
$Q(s,a)$ changes only that entry. A neural network instead shares parameters
across states, so one update can improve predictions for similar states that
were not in the sampled transition.

For a discrete action space $\mathcal{A}$, DQN usually takes a state as input
and produces one value per action:

$$
\widehat{Q}_\phi(s,\cdot)
\in\mathbb{R}^{|\mathcal{A}|}.
$$

The value for a sampled action is selected from this output vector. The greedy
action is the index of its largest entry:

$$
a^*(s)=\arg\max_{a\in\mathcal{A}}\widehat{Q}_\phi(s,a).
$$

Parameter sharing provides generalization, but it also couples the estimates.
Changing $\phi$ to improve one state-action pair can change predictions for
many other pairs. DQN needs two stabilizers to make these coupled Bellman
updates practical: a replay buffer and a target network.

!!! note "DQN assumes a discrete action space"
    DQN evaluates every action and takes a maximum over the resulting output
    vector. This is practical when $\mathcal{A}$ is finite and not too large.
    A continuous action space would require solving an optimization problem
    over actions for every target. Algorithms such as
    [SAC](../actor-critic/sac.md) use an actor for that
    purpose.

## Stabilizer 1: replay buffer

Consecutive environment transitions are strongly correlated. Training on each
transition immediately would present the network with many similar examples in
the order they occurred.

DQN stores transitions $(s,a,s',r,d)$ in a replay buffer $\mathcal{R}$ and
samples shuffled mini-batches from it. Replay has three effects:

- It reduces the temporal correlation within a training batch.
- It reuses a transition for more than one gradient update.
- It mixes data collected by older behavior policies with recent data.

The last point is valid because Q-learning is off-policy. Its target uses a
greedy next action regardless of which behavior policy produced the stored
action.

Replay cannot create coverage. If the behavior policy never tries an action in
a relevant state, the buffer contains no evidence from which to learn its
value. The [Q-learning overview](index.md#what-kind-of-data-should-we-collect)
describes $\epsilon$-greedy and Boltzmann exploration.

## Stabilizer 2: target network

Using one network for both sides of the regression would give the target

$$
y_i
=\mathbf{r}_i
+\gamma(1-\mathbf{d}_i)
\max_{a'}\widehat{Q}_\phi(\mathbf{s}'_i,a').
$$

The same parameters $\phi$ would determine both the prediction being fitted
and its label. A gradient update would change
$\widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)$, but it could also immediately
change the target through $\widehat{Q}_\phi(\mathbf{s}'_i,a')$. The network
would be chasing a label that moves with every update.

DQN keeps a target copy of the critic with parameters $\bar\phi$. The online
network $\widehat{Q}_\phi$ is optimized, while the target network
$\widehat{Q}_{\bar\phi}$ constructs the bootstrap target:

$$
y_i
=\mathbf{r}_i
+\gamma(1-\mathbf{d}_i)
\max_{a'}\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a').
$$

The online network minimizes

$$
\mathcal{L}(\phi)
=\frac{1}{2B}
\sum_{i=1}^{B}
\left(
\widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)-y_i
\right)^2.
$$

Detach every $y_i$ before this update. Gradients should change $\phi$ only;
they should not pass through the target into $\bar\phi$.

The original DQN update copies the online parameters into the target network
after every $C$ gradient steps:

$$
\bar\phi\leftarrow\phi.
$$

Here, $C$ is the target-update interval. Between copies, the targets remain
fixed even as the online network takes multiple gradient steps. A common
alternative is a soft update after every gradient step:

$$
\bar\phi
\leftarrow
\tau\phi+(1-\tau)\bar\phi,
$$

where $\tau$ is a small interpolation coefficient. Both methods delay how
quickly a critic update can change its own future targets.

## A full DQN algorithm walkthrough

DQN combines an online critic, a target critic, a replay buffer, and an
$\epsilon$-greedy behavior policy in one loop.

1. **Initialize the networks and buffer.** Initialize the online parameters
   $\phi$, set $\bar\phi\leftarrow\phi$, and create an empty replay buffer
   $\mathcal{R}$.
2. **Collect one transition.** At state $s$, take a uniformly random action
   with probability $\epsilon$ and take
   $a\in\arg\max_{a'}\widehat{Q}_\phi(s,a')$ otherwise. Execute $a$, observe
   $(s',r,d)$, and store $(s,a,s',r,d)$ in $\mathcal{R}$.
3. **Sample a batch.** Draw

    $$
    \left\{
    \left(
    \mathbf{s}_i,
    \mathbf{a}_i,
    \mathbf{s}'_i,
    \mathbf{r}_i,
    \mathbf{d}_i
    \right)
    \right\}_{i=1}^{B}
    \sim\mathcal{R}.
    $$

4. **Build fixed targets.** For every sampled transition, compute

    $$
    y_i
    =\mathbf{r}_i
    +\gamma(1-\mathbf{d}_i)
    \max_{a'}\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a'),
    $$

    and detach $y_i$ from the computation graph.

5. **Update the online critic.** Take a gradient step on

    $$
    \mathcal{L}(\phi)
    =\frac{1}{2B}
    \sum_{i=1}^{B}
    \left(
    \widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)-y_i
    \right)^2.
    $$

    The target parameters $\bar\phi$ do not change in this step.

6. **Update the target network.** Every $C$ gradient steps, copy
   $\bar\phi\leftarrow\phi$. If soft updates are used instead, move
   $\bar\phi$ slightly toward $\phi$ after every gradient step.
7. **Continue interacting.** Set $s\leftarrow s'$, or reset the environment if
   $d=1$. Return to step 2, keep adding transitions to $\mathcal{R}$, and
   gradually reduce $\epsilon$.

At evaluation time, set $\epsilon=0$ and act greedily with the online network.
The target network is needed for training targets, not for choosing evaluation
actions.

The target above uses one observed reward before bootstrapping. [N-step
returns](n-step-returns.md) use several rewards, which can propagate delayed
rewards faster but introduces a larger off-policy mismatch.

## Overestimation bias

The maximum in the DQN target prefers actions whose estimates happen to be too
large. Even if every action-value estimate has zero-mean error, taking their
maximum can introduce a positive bias:

$$
\mathbb{E}
\left[
\max_a\widehat{Q}(s,a)
\right]
\geq
\max_a
\mathbb{E}
\left[
\widehat{Q}(s,a)
\right].
$$

Standard DQN uses the target network both to select the maximizing action and
to evaluate it:

$$
a_i^*
=\arg\max_{a'}
\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a'),
\qquad
y_i
=\mathbf{r}_i
+\gamma(1-\mathbf{d}_i)
\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a_i^*).
$$

An action whose target-network estimate is too high is more likely to win the
maximum, and that same high estimate is then copied into the target. A target
network slows down target changes, but it does not separate action selection
from action evaluation.

[Double DQN](double-dqn.md) reduces this bias by using the online network to
select the greedy action and the target network to evaluate it.

## Practical details

- Fill the replay buffer with some transitions before starting gradient
  updates. Otherwise, early batches contain little variety.
- Treat $d=1$ as a true terminal state. A time-limit truncation usually still
  has future value and should continue to bootstrap.
- Normalize observations when their components have very different scales.
  Reward clipping can stabilize targets, but it changes the objective being
  optimized.
- DQN learns only from states and actions represented in the replay buffer.
  A larger network or buffer does not repair missing exploration coverage.

## Questions

See the [Q-Learning Q&A](q-and-a.md) for conceptual questions about off-policy
learning and target networks.
