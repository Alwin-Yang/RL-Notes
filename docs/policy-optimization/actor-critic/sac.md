# Soft Actor-Critic

Soft actor-critic (SAC) is an off-policy actor-critic algorithm. It reuses old
experience through a replay buffer and trains the policy to prefer both high
return and high entropy. We will build these two ideas separately.

## Replay buffers

The [PPO walkthrough](ppo.md#review-actor-critic-with-multiple-actor-updates)
discards each batch after a limited number of updates. Environment interaction
is often the slowest part of reinforcement learning, so a natural next step is
to keep the data and reuse it.

A **replay buffer** $\mathcal{D}$ stores transitions from past interactions.
Because PPO needs to know which policy collected each action, also store that
action's probability under the behavior policy $\mu$:

$$
(s_t,a_t,r_t,s_{t+1},d_t,\mu(a_t\mid s_t)).
$$

Here $s_t$ is the current state, $a_t$ is the action taken, $r_t$ is the reward,
$s_{t+1}$ is the resulting next state, and $d_t$ is one if the transition ends
the episode and zero otherwise. The behavior policy $\mu$ may be a different
older policy for every transition.

### PPO with a replay buffer

If we add the buffer directly to the PPO walkthrough, the loop becomes:

1. **Collect data.** Run the current actor in the environment. Append every
   transition and its action probability to $\mathcal{D}$ instead of replacing
   the previous batch.
2. **Sample replay data.** Draw a shuffled mini-batch
   $\{(\mathbf{s}_i,\mathbf{a}_i,\mathbf{r}_i,\mathbf{s}'_i,\mathbf{d}_i,
   \mu(\mathbf{a}_i\mid\mathbf{s}_i))\}_{i=1}^{B}$ from $\mathcal{D}$, where
   $B$ is the mini-batch size.
3. **Fit the critic.** Build the one-step target

    $$
    y_i
    =
    \mathbf{r}_i
    +
    \gamma(1-\mathbf{d}_i)V_\phi^\pi(\mathbf{s}'_i),
    $$

    then regress $V_\phi^\pi(\mathbf{s}_i)$ toward $y_i$.

4. **Evaluate advantages.** Compute

    $$
    \widehat{A}_i
    =
    \mathbf{r}_i
    +
    \gamma(1-\mathbf{d}_i)V_\phi^\pi(\mathbf{s}'_i)
    -
    V_\phi^\pi(\mathbf{s}_i).
    $$

5. **Update the actor.** Use the stored behavior probability to form

    $$
    \rho_i(\theta)
    =
    \frac{
    \pi_\theta(\mathbf{a}_i\mid\mathbf{s}_i)
    }{
    \mu(\mathbf{a}_i\mid\mathbf{s}_i)
    },
    $$

    then apply the PPO clipped update with $\rho_i(\theta)\widehat{A}_i$.

6. **Keep the data.** Return to step 1 without emptying $\mathcal{D}$.

This looks like PPO with better sample efficiency, but two parts of the update
now use data from policies that may be much older than the current actor.

### What breaks?

First, PPO clipping assumes that the behavior policy is close to the current
policy. In a replay buffer, $\mu$ may be many updates old. The importance ratio
can then be extremely large or small. Clipping prevents a large update, but it
also makes many old actions provide little or no policy gradient. A one-step
action ratio also does not correct the different state distribution created by
an old behavior policy.

The critic has a more direct problem. $V^\pi(s_t)$ averages over an action
$a_t\sim\pi_\theta(\cdot\mid s_t)$, but the reward and next state in a replayed
transition came from an action sampled from $\mu(\cdot\mid s_t)$. Therefore,

$$
r_t+\gamma V_\phi^\pi(s_{t+1})
$$

is a sample of the outcome of the behavior policy's action. Regressing
$V_\phi^\pi(s_t)$ toward this target averages over $\mu$ rather than the current
actor $\pi_\theta$. We could correct the action mismatch with importance
sampling, but the same extreme ratios would make the critic update noisy.

### Replace $V$ with $Q$

Instead of correcting the stored action, condition the critic on it. Recall
that $Q^\pi(s_t,a_t)$ is the expected discounted return after taking $a_t$ in
$s_t$ and then following $\pi_\theta$:

$$
Q^\pi(s_t,a_t)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
\sum_{t'=t}^{T}
\gamma^{t'-t}r(s_{t'},a_{t'})
\;\middle|\;s_t,a_t
\right].
$$

Its Bellman equation is

$$
Q^\pi(s_t,a_t)
=
\mathbb{E}_{\substack{
s_{t+1}\sim P(\cdot\mid s_t,a_t)\\
a_{t+1}\sim\pi_\theta(\cdot\mid s_{t+1})
}}
\left[
r_t+\gamma Q^\pi(s_{t+1},a_{t+1})
\right].
$$

### How to estimate $Q$

The Bellman equation is an expectation over all possible next states and
actions. We cannot compute that expectation exactly, so we use replay
transitions to build one-sample temporal-difference targets:

1. **Sample transitions.** Draw
   $\{(\mathbf{s}_i,\mathbf{a}_i,\mathbf{r}_i,
   \mathbf{s}'_i,\mathbf{d}_i)\}_{i=1}^{B}$ from $\mathcal{D}$.
2. **Sample current-policy actions.** For every next state, draw

    $$
    \bar{\mathbf{a}}'_i
    \sim
    \pi_\theta(\cdot\mid\mathbf{s}'_i).
    $$

    The bar distinguishes this newly sampled action from the old action
    $\mathbf{a}_i$ stored in the buffer.

3. **Build the target.** Compute

    $$
    y_i
    =
    \mathbf{r}_i
    +
    \gamma(1-\mathbf{d}_i)
    Q_\phi^\pi(\mathbf{s}'_i,\bar{\mathbf{a}}'_i).
    $$

    Treat $y_i$ as a constant during this update. If $\mathbf{d}_i=1$, the
    transition ended the episode, so the target contains only the observed
    reward.

4. **Fit the critic.** Minimize the mean squared Bellman error

    $$
    \mathcal{L}_Q(\phi)
    =
    \frac{1}{2B}
    \sum_{i=1}^{B}
    \left(
    Q_\phi^\pi(\mathbf{s}_i,\mathbf{a}_i)-y_i
    \right)^2.
    $$

The prediction on the left evaluates the stored state-action pair. The target
on the right uses the observed reward and next state, followed by a fresh action
from the current policy. Repeating this regression propagates later rewards
backward through the $Q$-function.

Once we condition on $(s_t,a_t)$, the reward and next state do not depend on
which policy selected $a_t$. This is why the stored transition remains useful
without an importance ratio. The target is still moving because it contains
the same learned critic; a later version will stabilize it with a target
network.

The actor also stops reusing the old action. For each replayed state $s_t$, it
samples a fresh action from the current policy and changes $\theta$ to increase
$Q_\phi^\pi(s_t,a_t)$. The stored action trains the critic; the fresh action
trains the actor. Neither update needs the PPO importance ratio.

This replacement gives SAC its off-policy actor-critic backbone. It is not yet
the final algorithm: the critic still needs stabilization, and the actor still
has no entropy objective.

!!! note "The buffer needs enough action coverage"
    Replacing $V$ with $Q$ removes the behavior-policy mismatch only for
    state-action pairs represented in the buffer. If $\mathcal{D}$ contains
    little or no data for an action near state $s$, then
    $Q_\phi^\pi(s,a)$ is an unsupported extrapolation. The actor may exploit
    this error by choosing an uncovered action that the critic incorrectly
    scores highly.

    The behavior policies must therefore explore enough that $\mathcal{D}$
    covers the actions the current actor may choose in relevant states. In a
    continuous action space, this does not mean sampling every possible action.
    It means collecting enough nearby actions for the critic to learn the
    relevant region reliably. Replay can reuse collected transitions, but it
    cannot supply evidence for actions that were never explored.

## Tricks used in SAC

The algorithm above can reuse replay data, but it is not yet SAC. SAC changes
the objective and adds several mechanisms that make the actor and critic stable
enough to train together.

### Trick 1: Add entropy to the objective

SAC maximizes both reward and policy entropy:

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
\sum_{t=1}^{T}
\gamma^{t-1}
\left(
r_t+\alpha\mathcal{H}
\left(\pi_\theta(\cdot\mid s_t)\right)
\right)
\right].
$$

Here

$$
\mathcal{H}\left(\pi_\theta(\cdot\mid s)\right)
=
-\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
\left[\log\pi_\theta(a\mid s)\right]
$$

is the policy entropy, and the temperature $\alpha>0$ controls the strength of
the entropy bonus. A larger $\alpha$ favors a more random policy; a smaller
$\alpha$ gives more weight to the critic.

With this objective, the next-state value is the expected $Q$-value plus the
entropy bonus:

$$
V^\pi(s')
=
\mathbb{E}_{a'\sim\pi_\theta(\cdot\mid s')}
\left[
Q^\pi(s',a')
-\alpha\log\pi_\theta(a'\mid s')
\right].
$$

The critic target therefore becomes

$$
y
=
r
+
\gamma(1-d)
\left[
Q^\pi(s',a')
-\alpha\log\pi_\theta(a'\mid s')
\right],
\qquad
a'\sim\pi_\theta(\cdot\mid s').
$$

The actor minimizes the corresponding loss

$$
\mathcal{L}_\pi(\theta)
=
\mathbb{E}_{s\sim\mathcal{D},\,
a\sim\pi_\theta(\cdot\mid s)}
\left[
\alpha\log\pi_\theta(a\mid s)-Q_\phi^\pi(s,a)
\right].
$$

Minimizing this loss makes high-value actions more likely while preventing the
policy from becoming deterministic too early.

### Trick 2: Use target critics

The critic target should not move immediately with the critic being optimized.
SAC keeps a target copy with parameters $\bar\phi$ and evaluates the bootstrap
term with $Q_{\bar\phi}$. After each update, it slowly moves the target
parameters toward the online parameters:

$$
\bar\phi
\leftarrow
\tau\phi+(1-\tau)\bar\phi,
$$

where $\tau$ is a small interpolation coefficient. This delayed target reduces
the feedback loop in which a critic update immediately changes its own
regression target.

### Trick 3: Use two critics

An actor searches for actions that the critic scores highly. It can therefore
exploit positive approximation errors in the critic even when those actions
are not actually good. SAC trains two independent critics,
$Q_{\phi_1}$ and $Q_{\phi_2}$, and uses the smaller prediction:

$$
Q_{\min}(s,a)
=
\min_{j\in\{1,2\}}Q_{\phi_j}(s,a).
$$

The minimum introduces a pessimistic bias, but reduces the more damaging
optimistic bias. SAC uses target copies of both critics when constructing the
bootstrap target.

### Trick 4: Reparameterize the actor

The actor must receive a gradient through its sampled action. For a Gaussian
policy, SAC writes the random action as a deterministic function of separate
noise:

$$
\begin{aligned}
\epsilon
&\sim\mathcal{N}(0,I),\\
u
&=
\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,\\
a
&=
\tanh(u).
\end{aligned}
$$

The noise $\epsilon$ does not depend on $\theta$, so gradients can pass from
$Q_\phi(s,a)$ through $a$ into the actor parameters. The $\tanh$ function keeps
each action coordinate in $(-1,1)$.

!!! warning "Correct the log probability after $\tanh$"
    The actor loss needs the density of the squashed action, not the original
    Gaussian density. Implementations subtract the change-of-variables term

    $$
    \log\pi_\theta(a\mid s)
    =
    \log\mathcal{N}
    \left(u;\mu_\theta(s),\sigma_\theta(s)\right)
    -
    \sum_k\log\left(1-\tanh^2(u_k)+\varepsilon\right),
    $$

    where $\varepsilon$ is a small numerical constant. Omitting this correction
    gives the entropy term the wrong gradient.

### Trick 5: Tune the temperature automatically

A fixed $\alpha$ requires task-specific tuning. SAC can instead learn it by
choosing a target entropy $\mathcal{H}_{\mathrm{target}}$ and minimizing

$$
\mathcal{L}_\alpha(\alpha)
=
\mathbb{E}_{\substack{
s\sim\mathcal{D}\\
a\sim\pi_\theta(\cdot\mid s)
}}
\left[
\alpha
\left(
-\log\pi_\theta(a\mid s)
-\mathcal{H}_{\mathrm{target}}
\right)
\right].
$$

If the policy entropy is below the target, gradient descent increases
$\alpha$, which puts more weight on randomness. If the entropy is above the
target, it decreases $\alpha$. Implementations usually optimize
$\log\alpha$ so that $\alpha$ remains positive.

## A full SAC algorithm walkthrough

SAC combines the replay buffer and all five tricks in one loop. Three different
actions appear in an update, so we give each one separate notation:

- $\mathbf{a}_i^{\mathcal{D}}$ is the old action stored in the replay buffer.
- $\bar{\mathbf{a}}'_i$ is sampled now from the current policy at the buffered
  next state $\mathbf{s}'_i$.
- $\widetilde{\mathbf{a}}_i$ is sampled now from the current policy at the
  buffered current state $\mathbf{s}_i$.

Only the first action comes from the buffer. The other two are new policy
samples and generally differ from every action stored in $\mathcal{D}$.

!!! note "The states are not sampled from $p_\theta$"
    The states $\mathbf{s}_i$ and $\mathbf{s}'_i$ are sampled from the replay
    buffer, not from a fresh trajectory $\tau\sim p_\theta$ generated by the
    current policy. If $d^{\pi_\theta}(s)$ denotes the current policy's
    discounted state-visitation distribution, then SAC uses
    $s\sim\mathcal{D}$ where an on-policy policy gradient would use
    $s\sim d^{\pi_\theta}$.

    We accept this mismatch for the critic because the Bellman equation holds
    separately at each state-action pair. With enough coverage and an exact
    function approximator, changing the distribution used to sample Bellman
    updates changes how often each pair is trained, but not the Bellman fixed
    point.

    The actor claim is weaker. Sampling states from $\mathcal{D}$ means that
    its loss is a replay-weighted surrogate, not the exact current-policy
    objective. SAC accepts this bias because it allows many low-variance
    updates per environment transition. In practice, the buffer is continually
    refreshed, the policy usually changes gradually, and recent replay states
    remain relevant. If the buffer is stale or misses states reached by the
    current policy, this approximation can fail.

1. **Initialize the networks and buffer.** Create the actor $\pi_\theta$, two
   critics $Q_{\phi_1}$ and $Q_{\phi_2}$, their target copies
   $Q_{\bar\phi_1}$ and $Q_{\bar\phi_2}$, the temperature $\alpha$, and an
   empty replay buffer $\mathcal{D}$.
2. **Interact with the environment.** At the current environment state $s_t$,
   sample $a_t\sim\pi_\theta(\cdot\mid s_t)$, execute it, and store
   $(s_t,a_t,r_t,s_{t+1},d_t)$ in $\mathcal{D}$. The action is sampled from the
   current policy at interaction time; it becomes a buffer action only after
   it is stored. SAC does not need to store its old behavior probability.
3. **Sample transitions from the buffer.** Draw

    $$
    \left\{
    \left(
    \mathbf{s}_i,
    \mathbf{a}_i^{\mathcal{D}},
    \mathbf{r}_i,
    \mathbf{s}'_i,
    \mathbf{d}_i
    \right)
    \right\}_{i=1}^{B}
    \sim\mathcal{D}.
    $$

    All five quantities in this step come from the replay buffer.

4. **Sample next actions from the current policy.** For every buffered next
   state, sample

    $$
    \bar{\mathbf{a}}'_i
    \sim
    \pi_\theta(\cdot\mid\mathbf{s}'_i).
    $$

    The state $\mathbf{s}'_i$ comes from the buffer, but
    $\bar{\mathbf{a}}'_i$ and its log probability are computed now using the
    current actor. They are not read from the transition that followed
    $\mathbf{s}'_i$ in an old trajectory.

5. **Build the soft critic target.** Compute

    $$
    \begin{aligned}
    y_i
    &=
    \mathbf{r}_i
    +
    \gamma(1-\mathbf{d}_i)
    \left[
    \min_{j\in\{1,2\}}
    Q_{\bar\phi_j}
    (\mathbf{s}'_i,\bar{\mathbf{a}}'_i)
    -
    \alpha
    \log\pi_\theta
    (\bar{\mathbf{a}}'_i\mid\mathbf{s}'_i)
    \right].
    \end{aligned}
    $$

    The reward, terminal flag, and next state come from the buffer. The next
    action and log probability come from the current policy. The $Q$-values
    come from the target critics. Detach the entire target $y_i$ before the
    critic update.

6. **Update both critics on the buffered actions.** For each
   $j\in\{1,2\}$, minimize

    $$
    \mathcal{L}_{Q_j}(\phi_j)
    =
    \frac{1}{2B}
    \sum_{i=1}^{B}
    \left(
    Q_{\phi_j}
    (\mathbf{s}_i,\mathbf{a}_i^{\mathcal{D}})
    -
    y_i
    \right)^2.
    $$

    This is the only network update that uses the old buffered action
    $\mathbf{a}_i^{\mathcal{D}}$.

7. **Sample new actor actions.** For every buffered current state, use the
   reparameterized actor to sample

    $$
    \widetilde{\mathbf{a}}_i
    \sim
    \pi_\theta(\cdot\mid\mathbf{s}_i).
    $$

    The state comes from the buffer, but the action and its log probability
    come from the current policy. Do not use
    $\mathbf{a}_i^{\mathcal{D}}$ in the actor loss.

8. **Update the actor.** Minimize

    $$
    \mathcal{L}_\pi(\theta)
    =
    \frac{1}{B}
    \sum_{i=1}^{B}
    \left[
    \alpha
    \log\pi_\theta
    (\widetilde{\mathbf{a}}_i\mid\mathbf{s}_i)
    -
    \min_{j\in\{1,2\}}
    Q_{\phi_j}
    (\mathbf{s}_i,\widetilde{\mathbf{a}}_i)
    \right].
    $$

    Gradients pass through the reparameterized action into $\theta$. The
    critics supply gradients with respect to the action, but their parameters
    are not updated by the actor optimizer.

9. **Update the temperature.** Use the same current-policy log probabilities
   to move the entropy toward $\mathcal{H}_{\mathrm{target}}$:

    $$
    \mathcal{L}_\alpha(\alpha)
    =
    \frac{1}{B}
    \sum_{i=1}^{B}
    \alpha
    \left[
    -\log\pi_\theta
    (\widetilde{\mathbf{a}}_i\mid\mathbf{s}_i)
    -
    \mathcal{H}_{\mathrm{target}}
    \right].
    $$

    Treat the log probabilities as constants in this step. The temperature
    optimizer changes $\alpha$, not the actor.

10. **Update the target critics.** For $j\in\{1,2\}$, apply

    $$
    \bar\phi_j
    \leftarrow
    \tau\phi_j+(1-\tau)\bar\phi_j.
    $$

11. **Keep the replay data and repeat.** Continue interacting with the
    environment and taking gradient updates without emptying $\mathcal{D}$.

!!! warning "Termination and truncation are different"
    Set $d_t=1$ when the environment reaches a true terminal state. A time-limit
    truncation does not imply that the underlying state has no future value, so
    implementations should usually continue bootstrapping across it.

## SAC summary

SAC is an off-policy maximum-entropy actor-critic algorithm. The replay buffer
provides states, old actions, rewards, next states, and terminal flags. The old
actions train the critics, because each replayed reward and next state belongs
to that recorded state-action pair.

The actor update does not imitate those old actions. It samples new actions from
the current policy at replayed states and maximizes their pessimistic predicted
value plus entropy. The critic target likewise samples a new next action from
the current policy rather than reading the next action from the buffer. This
separation is what lets SAC reuse old transitions without PPO importance
ratios.

Twin critics reduce optimistic value errors, target critics stabilize
bootstrapping, reparameterization gives the actor a low-variance gradient, and
automatic temperature tuning controls the reward-entropy trade-off. SAC is
sample-efficient because it reuses experience, but it still depends on enough
state-action coverage in the replay buffer.
