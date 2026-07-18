# Policy Optimization Q&A
<div class="qa-list" markdown>


## Why policy optimization use gradient ascent?

Because we want to maximize the expected return, and gradient ascent is a method for finding the maximum of a function.


## Why are policy gradients high variance?

They estimate expected return from sampled trajectories, whose rewards and actions can vary substantially.





## What does PPO clipping do?

It limits the incentive to move the new policy too far from the policy that collected the rollout.
</div>
