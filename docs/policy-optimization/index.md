# Policy Optimization

Policy methods optimize a parameterised policy directly and work naturally with continuous action spaces.

## 1. Define the policy objective

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)].$$

The objective is the expected return over trajectories sampled from the current policy. Policy optimization therefore performs gradient ascent on $J(\theta)$.

## 2. Estimate the policy gradient

Because the expectation cannot usually be computed exactly, REINFORCE samples trajectories and uses their returns to estimate $\nabla_\theta J(\theta)$.

This estimate is unbiased but can have high variance: both the policy and the environment may be stochastic, so different trajectories can produce very different returns.

## 3. Improve the gradient estimate

[Policy-gradient improvements](policy-improvement.md) make learning more stable without changing the expected gradient:

1. **REINFORCE** weights every action in a trajectory by the trajectory's total return.
2. **Reward-to-go** removes rewards that occurred before an action, using causality to reduce variance.
3. **A baseline** removes additional variance while keeping the estimator unbiased.

The progression is therefore not a sequence of different objectives; it is a sequence of increasingly useful estimators of the same policy gradient.

## 4. Constrain the policy update

PPO constrains policy updates through a clipped surrogate objective.

After estimating the gradient, it is still important to control the update size: a large neural-network update can change the policy sharply and invalidate the data collected by the previous policy.

## Questions

See the [Policy Optimization Q&A](q-and-a.md) for conceptual questions about gradient ascent, gradient variance, rollout batches, and PPO clipping.
