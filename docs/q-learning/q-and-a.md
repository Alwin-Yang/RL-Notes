# Q-Learning Q&A

<div class="qa-list" markdown>

## Why is Q-learning off-policy?

Its target uses the greedy next action, regardless of the behaviour policy
that collected the transition.

## Why does DQN need a target network?

Keeping the bootstrap target temporarily fixed reduces feedback loops during
optimization.

## Why are n-step Q-learning targets harder to use off-policy?

Their intermediate rewards and states depend on actions from the behavior
policy, so a greedy bootstrap at the final state does not correct the whole
sampled continuation.
</div>
