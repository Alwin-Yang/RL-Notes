# Policy Optimization

Policy methods optimize a parameterised policy directly and work naturally with continuous action spaces.

## Policy gradients

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)].$$

## Proximal Policy Optimization

PPO constrains policy updates through a clipped surrogate objective.

### Notes to add

- REINFORCE, baselines, and advantage estimation
- Clipping range, rollout length, and entropy regularisation
