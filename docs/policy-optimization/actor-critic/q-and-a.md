# Actor-Critic Q&A

<div class="qa-list" markdown>

## What is the critic for?

It supplies a value estimate that acts as a baseline or advantage estimator for the actor.

## Why does an on-policy actor-critic learn $V^\pi$ instead of $Q^\pi$?

Because the advantage can be built from $V^\pi$ alone: the observed reward and
next state already give $\delta_t=r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)$, whereas
recovering $V^\pi$ from $Q^\pi$ needs an average over the action space and
bootstrapping $Q^\pi$ needs a next action as well. $V^\pi$ is also defined only
on states, so the same data covers a much smaller input space than
$\mathcal{S}\times\mathcal{A}$.

## When is learning $~Q^\pi ~$ the better choice?

When the value function has to support off-policy updates or produce the policy
itself: $Q^\pi$ can be trained on replay data and lets us act by maximizing over
actions, as in DQN, or update an actor by differentiating through the critic, as
in DDPG, TD3, and SAC. The price is that maximizing over actions systematically
overestimates values, which is why those methods add corrections such as twin
critics.

## Is learning a $~Q^\pi ~$ critic the same as Q-learning?

No, and the difference is how the target treats the next action. A critic
evaluates the current policy, so its target takes an expectation over that
policy, $r_t+\gamma Q^\pi(s_{t+1},a_{t+1})$ with $a_{t+1}\sim\pi_\theta$, while
Q-learning uses $r_t+\gamma\max_{a'}Q(s_{t+1},a')$ and therefore converges to the
optimal $Q^*$ instead of to $Q^\pi$. The $\max$ also makes the actor unnecessary,
since the policy is just $\arg\max_a Q(s,a)$, which is why Q-learning is a
value-based method rather than a variant of actor-critic. Methods like DDPG and
SAC sit in between: the $\max$ is intractable over continuous actions, so the
actor is trained to approximate it.

## Why does SAC use two critics?

Taking the smaller target value reduces optimistic value-estimation bias.
</div>
