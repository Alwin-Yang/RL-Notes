# Deep Q-Networks

A deep Q-network (DQN) replaces the Q-table in
[Q-learning](index.md) with a neural network $Q_\phi(s,a)$, whose parameters
are $\phi$. This allows Q-learning to handle state spaces where storing one
value for every state-action pair is impractical.

## The DQN target

For a transition $(s_t,a_t,r_t,s_{t+1})$, DQN uses the target

$$
y_t = r_t + \gamma \max_{a'} Q_{\phi^-}(s_{t+1},a'),
$$

where $\phi^-$ are the target-network parameters. The online network is trained
to reduce the squared temporal-difference error

$$
\left(y_t-Q_\phi(s_t,a_t)\right)^2.
$$

## Why DQN needs two stabilizers

DQN adds two mechanisms to ordinary Q-learning:

- A replay buffer stores transitions and samples them in shuffled mini-batches.
  This reduces correlation between consecutive training examples and allows
  each transition to be reused.
- A target network holds $\phi^-$ fixed for several updates. This prevents the
  regression target from changing after every update to the online network.

Without these mechanisms, a neural network is fitted to targets that move
rapidly and are computed from strongly correlated data. Training can become
unstable or diverge.

### Notes to add

- Exploration schedules and observation scaling
- Overestimation bias and Double DQN
