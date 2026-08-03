# Double DQN

Double DQN modifies one part of [DQN](dqn.md): how the bootstrap target chooses
and evaluates the next action. It uses the same online network, target network,
replay buffer, and $\epsilon$-greedy data collection as DQN.

## Why the DQN maximum is biased

Suppose the estimated Q-value of every action equals its true value plus some
approximation error. Even when each error has zero mean, taking a maximum tends
to select an action with a positive error:

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

Standard DQN constructs the target with

$$
y_i^{\mathrm{DQN}}
=\mathbf{r}_i
+\gamma(1-\mathbf{d}_i)
\max_{a'}
\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a').
$$

The maximum performs two jobs with the same target-network predictions:

1. It selects the action with the largest estimated value.
2. It uses that same estimate as the value of the selected action.

If an approximation error makes one action look too good, that action is more
likely to be selected, and its overestimated value is then used in the target.
This is overestimation bias.

## Why the target network is not enough

The target network holds $\bar\phi$ fixed while the online network is updated.
This stabilizes the regression target, but it does not address the maximum's
selection bias. The target network still selects and evaluates the action using
the same set of noisy predictions.

Double DQN assigns those two jobs to different networks.

## Separate selection from evaluation

First, use the online network to select the greedy action at the buffered next
state:

$$
a_i^*
=\arg\max_{a'}
\widehat{Q}_\phi(\mathbf{s}'_i,a').
$$

Then use the target network to evaluate only that action:

$$
y_i^{\mathrm{Double}}
=\mathbf{r}_i
+\gamma(1-\mathbf{d}_i)
\widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a_i^*).
$$

The online network may select an action because its estimate contains a
positive error. The target network does not necessarily make the same error on
that action, so the positive selection error is less likely to be copied into
the target. The networks are related because $\bar\phi$ is periodically copied
from $\phi$, so Double DQN reduces overestimation but does not guarantee that
it disappears.

!!! note "Double DQN does not add another network"
    DQN already has an online network and a target network. Double DQN reuses
    those two networks and changes their roles in the target. It does not train
    two independent critics.

## A Double DQN algorithm walkthrough

The DQN training loop remains unchanged except for target construction:

1. **Collect data.** Use an $\epsilon$-greedy behavior policy based on
   $\widehat{Q}_\phi$ and store transitions in the replay buffer
   $\mathcal{R}$.
2. **Sample a batch.** Draw
   $\{(\mathbf{s}_i,\mathbf{a}_i,\mathbf{s}'_i,\mathbf{r}_i,
   \mathbf{d}_i)\}_{i=1}^{B}$ from $\mathcal{R}$.
3. **Select next actions with the online network.** Compute

    $$
    a_i^*
    =\arg\max_{a'}
    \widehat{Q}_\phi(\mathbf{s}'_i,a').
    $$

4. **Evaluate them with the target network.** Build

    $$
    y_i^{\mathrm{Double}}
    =\mathbf{r}_i
    +\gamma(1-\mathbf{d}_i)
    \widehat{Q}_{\bar\phi}(\mathbf{s}'_i,a_i^*).
    $$

    Treat both the selected action $a_i^*$ and the target
    $y_i^{\mathrm{Double}}$ as fixed during the critic update.

5. **Update the online network.** Minimize

    $$
    \mathcal{L}(\phi)
    =\frac{1}{2B}
    \sum_{i=1}^{B}
    \left(
    \widehat{Q}_\phi(\mathbf{s}_i,\mathbf{a}_i)
    -y_i^{\mathrm{Double}}
    \right)^2.
    $$

6. **Update the target network and repeat.** Copy or slowly move $\bar\phi$
   toward $\phi$ using the same schedule as DQN, then continue collecting data
   and training.

The key distinction is the source of each quantity:

- The buffered current action $\mathbf{a}_i$ comes from an old behavior policy.
- The selected next action $a_i^*$ comes from the current online network.
- The value of $a_i^*$ comes from the target network.

## What Double DQN does not solve

Double DQN addresses overestimation caused by using one estimator for both
selection and evaluation. It still needs replay, a target-update schedule, and
adequate exploration. Function approximation, incomplete state-action
coverage, and optimization error can still produce inaccurate Q-values.

## Questions

See the [Q-Learning Q&A](q-and-a.md) for conceptual questions about off-policy
learning and target networks.
