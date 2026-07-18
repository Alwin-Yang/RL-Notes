# Policy Optimization Q&A
<div class="qa-list" markdown>


## Why policy optimization use gradient ascent?

Because we want to maximize the expected return, and gradient ascent is a method for finding the maximum of a function.


## Why are policy gradients high variance?

- Policy-gradient estimates are computed from a finite number of sampled trajectories.
- Actions, state transitions, and rewards can all be stochastic, so trajectories may differ substantially.
- The gradient is weighted by noisy, long-horizon returns, which amplifies sampling noise and makes credit assignment difficult.
- Sparse or large-scale rewards can further increase the variance.





## What does PPO clipping do?

It limits the incentive to move the new policy too far from the policy that collected the rollout.
</div>
