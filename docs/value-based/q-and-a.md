# Value-Based RL Q&A

<div class="qa-list" markdown>

## Why is Q-learning off-policy?

Its target uses the greedy next action, regardless of the behaviour policy that collected the transition.

## Why does DQN need a target network?

Keeping the bootstrap target temporarily fixed reduces feedback loops during optimization.
</div>
