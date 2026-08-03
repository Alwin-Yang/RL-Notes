# Q-Learning

Q-learning is a value-based method: it learns action values and derives the
policy from them rather than representing the policy with a separate actor.

## Recall: definition of the Q-function

For a fixed policy $\pi$, the action-value function $Q^\pi(s_t,a_t)$ is the
expected discounted return after taking action $a_t$ in state $s_t$ and then
following $\pi$:

$$
Q^\pi(s_t,a_t)
=\mathbb{E}
\left[
\sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'},a_{t'})
\;\middle|\;s_t,a_t
\right].
$$

## From actor-critic to critic only

Actor-critic methods represent the policy and the value function separately.
The actor chooses actions, while the critic evaluates them. If the critic
estimates $Q(s,a)$, however, it already tells us which action currently looks
best. We can therefore remove the separate actor and derive a greedy policy
directly from the critic:

$$
\pi(a\mid s)
=
\begin{cases}
1, & a=\arg\max_{a'}Q(s,a'),\\
0, & \text{otherwise}.
\end{cases}
$$

This is a critic-only method: the policy has no separate parameters and changes
whenever the Q-function changes. The expression assumes either a unique
maximizing action or a fixed rule for breaking ties.

## A full algorithm walkthrough

Everything above assembles into one loop over a parameterized critic
$\widehat{Q}_\phi(s,a)$ and no separate actor. A replay buffer $\mathcal{R}$
stores old transitions so that the critic can learn from a batch rather than
from only the latest transition. Initialize the critic parameters $\phi$ and an
empty replay buffer $\mathcal{R}$, then repeat the following steps:

1. **Collect data from $\pi$.** At the current state $s$, sample
   $a\sim\pi(\cdot\mid s)$, execute it, and observe the reward $r$, the next
   state $s'$, and a terminal indicator $d$. Store $(s,a,s',r,d)$ in
   $\mathcal{R}$.
2. **Sample a batch.** Draw $B$ transitions from the replay buffer:

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

3. **Build target Q-values.** For every sampled transition, use a one-step
   Bellman backup to estimate the Q-value that the critic should predict:

    $$
    y_i
    =\mathbf{r}_i
    +\gamma(1-\mathbf{d}_i)
    \max_{a'}\widehat{Q}_\phi(\mathbf{s}'_i,a').
    $$

    The factor $1-\mathbf{d}_i$ removes the bootstrap term at a terminal state.
    Treat $y_i$ as a fixed regression target during the critic update.

4. **Fit $\widehat{Q}_\phi$ to the target Q-values.** Update $\phi$ by
   minimizing the mean squared temporal-difference error:

    $$
    \mathcal{L}(\phi)
    =\frac{1}{2B}
    \sum_{i=1}^{B}
    \left(
    \widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)-y_i
    \right)^2.
    $$

5. **Improve the policy.** Derive a new greedy policy from the updated critic:

    $$
    \pi(a\mid s)
    =
    \begin{cases}
    1, & a=\arg\max_{a'}\widehat{Q}_\phi(s,a'),\\
    0, & \text{otherwise}.
    \end{cases}
    $$

    There is no actor loss or separate policy update. Changing
    $\widehat{Q}_\phi$ changes the greedy action and therefore changes the
    policy immediately.

6. **Continue interacting.** Set $s\leftarrow s'$. If the transition was
   terminal, reset the environment. Then return to step 1 and keep adding new
   transitions to $\mathcal{R}$ while training the critic on sampled batches.

### Why the target makes sense

The return from time $t$ can be split into the current reward and the return
from the next step:

$$
G_t=r_t+\gamma G_{t+1}.
$$

For a fixed policy $\pi$, substituting this identity into the definition of
$Q^\pi$ gives the Bellman expectation equation:

$$
Q^\pi(s,a)
=r(s,a)
+\gamma
\mathbb{E}_{\substack{s'\sim P(\cdot\mid s,a)\\
a'\sim\pi(\cdot\mid s')}}
\left[Q^\pi(s',a')\right].
$$

Here, $P(\cdot\mid s,a)$ is the environment's transition distribution and
$r(s,a)$ is the expected immediate reward. The equation averages the next
Q-value over both the next state and the next action selected by $\pi$.

Now define the optimal action-value function as the highest value available
over all policies:

$$
Q^*(s,a)=\max_\pi Q^\pi(s,a).
$$

After the environment produces $s'$, an optimal policy can choose whichever
next action has the highest Q-value. The expectation over $a'$ in the fixed
policy equation therefore becomes a maximum:

$$
Q^*(s,a)
=r(s,a)
+\gamma
\mathbb{E}_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'}Q^*(s',a')
\right].
$$

This is the Bellman optimality equation. The maximum is what distinguishes it
from the Bellman expectation equation for a fixed policy.

The transition distribution $P$ is unknown, so Q-learning replaces the
expectation with a sampled transition and replaces the unknown $Q^*$ with the
current approximation $\widehat{Q}_\phi$. This gives the target from step 3:

$$
y_i
=\mathbf{r}_i
+\gamma(1-\mathbf{d}_i)
\max_{a'}\widehat{Q}_\phi(\mathbf{s}'_i,a').
$$

Step 4 fits $\widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)$ to this target.
Repeating this update across many transitions pushes the learned Q-function
toward a fixed point:

$$
\widehat{Q}(s,a)
=\mathbb{E}
\left[
r+\gamma\max_{a'}\widehat{Q}(s',a')
\mid s,a
\right].
$$

For a discounted MDP with $\gamma<1$, the Bellman optimality operator has a
unique fixed point. Because $Q^*$ satisfies the same equation, that fixed point
is $Q^*$. With finite data, function approximation, and incomplete
optimization, $\widehat{Q}_\phi$ may not reach it exactly; the update only tries
to move the critic toward it.

## What kind of data should we collect?

The greedy policy always selects the action with the largest estimated
Q-value. Early in training, those estimates are inaccurate. Acting greedily
from the start can repeatedly select one action and leave the values of other
actions untested.

The walkthrough writes the data-collection policy as $\pi$ for simplicity. In
practice, it can be an exploratory behavior policy, which we denote by $\mu$.
The collected action comes from $\mu$, while the target uses a greedy next
action. The behavior and target policies can differ, which is why Q-learning
is off-policy.

The behavior policy should provide coverage: it should try multiple actions in
the states relevant to the task. A replay buffer can reuse collected
transitions, but it cannot provide information about actions that were never
taken. Two common exploration policies are $\epsilon$-greedy and Boltzmann
exploration.

### A shorter walkthrough with exploration

The full walkthrough above can be repeated with a separate behavior policy:

1. **Collect data from an exploration policy.** Sample
   $a\sim\mu(\cdot\mid s)$, execute it, and add the resulting transition to
   $\mathcal{R}$.
2. **Sample a batch.** Draw transitions
   $\{(\mathbf{s}_i,\mathbf{a}_i,\mathbf{s}'_i,\mathbf{r}_i,
   \mathbf{d}_i)\}_{i=1}^{B}$ from $\mathcal{R}$.
3. **Fit the critic.** Use the greedy target

    $$
    y_i
    =\mathbf{r}_i
    +\gamma(1-\mathbf{d}_i)
    \max_{a'}\widehat{Q}_\phi(\mathbf{s}'_i,a')
    $$

    and fit $\widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)$ to $y_i$.

4. **Update the data-collection policy.** The maximum in step 3 already uses
   the improved greedy policy. For the next round of data collection, define
   an $\epsilon$-greedy behavior policy $\mu$:

    $$
    a\sim\mu(\cdot\mid s),
    \qquad
    a=
    \begin{cases}
    \text{a uniformly random action}, & \text{with probability }\epsilon,\\
    \displaystyle\arg\max_{a'}\widehat{Q}_\phi(s,a'),
    & \text{with probability }1-\epsilon.
    \end{cases}
    $$

5. **Repeat.** Use the updated $\mu$ to collect more data and continue
   training.

The behavior policy $\mu$ determines which transitions enter the replay
buffer. The greedy policy $\pi$ remains the target policy represented by the
maximum in step 3.

### $\epsilon$-greedy exploration

With probability $\epsilon$, choose an action uniformly from the action space.
With probability $1-\epsilon$, choose a greedy action:

$$
a=
\begin{cases}
\text{a uniformly random action}, & \text{with probability }\epsilon,\\
\displaystyle\arg\max_{a'}\widehat{Q}_\phi(s,a'),
& \text{with probability }1-\epsilon.
\end{cases}
$$

A large $\epsilon$ gives more exploration. A small $\epsilon$ uses the current
Q-function more often. Training commonly starts with a larger $\epsilon$ and
reduces it over time, while evaluation uses $\epsilon=0$.

### Boltzmann exploration

Boltzmann exploration samples every action with probability proportional to
the exponential of its estimated Q-value:

$$
\mu(a\mid s)
=\frac{
\exp\left(\widehat{Q}_\phi(s,a)/\tau\right)
}{
\sum_{a'}\exp\left(\widehat{Q}_\phi(s,a')/\tau\right)
},
$$

where the temperature $\tau>0$ controls exploration. A high temperature makes
the action probabilities closer to uniform. As $\tau$ approaches zero, the
policy concentrates on greedy actions. We exponentiate the Q-values rather
than use them as probabilities directly because Q-values can be negative and
do not have to sum to one.

## DQN

[Deep Q-networks](dqn.md) implement $\widehat{Q}_\phi(s,a)$ with a neural
network. They also use a separate target network so that the bootstrap targets
do not change after every critic update.

## Questions

See the [Q-Learning Q&A](q-and-a.md) for conceptual questions about off-policy
learning and DQN target networks.
