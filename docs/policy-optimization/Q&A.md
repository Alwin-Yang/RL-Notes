# Policy Optimization Q&A
<div class="qa-list" markdown>


## Why policy optimization use gradient ascent?

Because we want to maximize the expected return, and gradient ascent is a method for finding the maximum of a function.


## Why are policy gradients high variance?

- Policy-gradient estimates are computed from a finite number of sampled trajectories.
- Actions, state transitions, and rewards can all be stochastic, so trajectories may differ substantially.
- The gradient is weighted by noisy, long-horizon returns, which amplifies sampling noise and makes credit assignment difficult.
- Sparse or large-scale rewards can further increase the variance.





## Why does an on-policy algorithm not update the neural network after every single rollout?

- In principle, the network can be updated after one rollout, but one rollout is only a noisy sample of the expected return.
- A stochastic policy can select different actions in the same state and therefore generate different trajectories.
- Even with a deterministic policy, the environment may still be stochastic because of random initial states, state transitions, or rewards.
- On-policy algorithms therefore usually keep the policy fixed while collecting multiple trajectories, then average their gradient estimates before updating the network.
- Using multiple trajectories produces a more reliable estimate of the expected policy gradient and makes training more stable.


## What does PPO clipping do?

It limits the incentive to move the new policy too far from the policy that collected the rollout.
</div>
