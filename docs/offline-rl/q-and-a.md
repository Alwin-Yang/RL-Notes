# Offline Reinforcement Learning Q&A

<div class="qa-list" markdown>

## Why can an online off-policy algorithm fail when trained on a fixed dataset?

Its actor can exploit overestimated values for actions that are absent from the
dataset, while the learner cannot collect new evidence to correct those values.

## Why is behavior cloning an important baseline for offline RL?

Behavior cloning avoids most out-of-distribution actions, so an offline RL
method that performs worse is likely being harmed by its value estimates or
policy updates.

## What breaks if the policy constraint is too weak?

The actor can move toward unsupported actions whose Q-values are unreliable,
which lets extrapolation error enter and grow through Bellman backups.

## What breaks if the policy constraint is too strong?

The learned policy stays close to the behavior policy and may be unable to
prefer the better actions in a mixed-quality dataset.

## Why does CQL lower values for actions outside the dataset?

Lower OOD values prevent the actor from treating critic extrapolation as real
evidence of high return.

## Why is making every Q-value very small not a valid form of conservatism?

The critic must still preserve useful value differences among supported
actions; otherwise the actor has no signal for policy improvement.

## How does IQL reduce queries on out-of-distribution actions?

It fits a state value from Q-values of dataset actions and uses that value in
the Bellman target instead of sampling a next action from the learned policy.

## Why can offline RL not overcome missing dataset coverage?

If the data contain no evidence about an action's consequences, no offline
objective can identify its value without additional assumptions.

## Why must time-limit truncation be separated from true termination?

A truncated state may still have future value and require bootstrapping, while
a true terminal state does not.

## Why can predicted Q-values be misleading during model selection?

The same critic errors that drive the policy can inflate its predicted value,
so a high estimate does not prove that the policy will obtain a high return.

## What makes importance sampling unreliable for long offline trajectories?

It multiplies policy probability ratios across time, which can produce a few
extreme weights and very high estimator variance.

</div>
