# Q-Learning

Q-learning estimates the optimal action-value function $Q^*$ and acts greedily
with respect to it. It is a value-based method: the policy is derived from the
learned values rather than represented by a separate actor.

## Tabular Q-learning

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right].
$$

Here, $\alpha$ is the learning rate, $r$ is the observed reward, and $\gamma$
is the discount factor. The maximum uses the greedy next action even if another
policy collected the transition. This makes Q-learning off-policy.

## From Q-learning to DQN

[Deep Q-networks](dqn.md) extend Q-learning to large state spaces by replacing
the table with a neural network. A replay buffer and a target network make the
resulting optimization more stable.

## Questions

See the [Q-Learning Q&A](q-and-a.md) for conceptual questions about off-policy
learning and DQN target networks.
