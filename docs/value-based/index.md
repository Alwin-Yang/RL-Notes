# Value-Based RL

Value-based methods estimate action quality, then act greedily with respect to the learned value function.

## Q-learning

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right].
$$

## Deep Q-networks

DQN combines Q-learning with a neural approximation, replay buffer, and target network.

### Notes to add

- Exploration schedules and overestimation bias
- Target networks, replay buffers, and observation scaling
