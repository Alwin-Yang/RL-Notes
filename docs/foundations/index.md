
# Foundations

The mathematical vocabulary used by most reinforcement-learning methods.

## Markov decision processes

An MDP is $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$: states, actions, transitions, rewards, and a discount factor.

## Dynamic programming

Repeated Bellman backups under a known environment model.

## Estimators

Every quantity an RL algorithm needs is estimated from samples, so it helps to
know when an estimate is right on average and when it is not. See
[Bias and Variance](bias-and-variance.md).

### Notes to add

- Returns, policies, and value functions
- Policy evaluation, policy iteration, and value iteration
