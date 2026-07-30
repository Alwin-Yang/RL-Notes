# Bias and Variance of Estimators

Almost nothing in reinforcement learning is computed exactly. Returns, value
functions, and policy gradients are all estimated from a finite number of
sampled interactions. Two properties then decide whether an algorithm works:
whether the estimate points at the right answer on average, and how far it
scatters around that average. The first is bias, the second is variance. This
page fixes both definitions on a plain statistics example, then names the places
where bias enters RL.

## What an estimator is

We want an unknown number $\theta$. We observe a random sample
$X_1,\dots,X_n$ and compress it into one number with some function $T$:

$$
\hat\theta = T(X_1,\dots,X_n).
$$

The point to hold on to is that $\hat\theta$ is itself a random variable:
another batch of samples gives another value. Its **bias** is the gap between
its mean and the truth,

$$
\mathrm{bias}(\hat\theta)=\mathbb{E}[\hat\theta]-\theta,
$$

where the expectation is taken over the sampling distribution and $\theta$ is a
constant. The estimator is **unbiased** when this gap is zero, and **biased**
otherwise. Its **variance** is $\mathrm{Var}(\hat\theta)$, the spread of
$\hat\theta$ around its own mean $\mathbb{E}[\hat\theta]$.

Unbiased means that the average over infinitely many repetitions lands on
$\theta$. It says nothing about any single draw. Keeping the two apart matters
in RL, where the reward-to-go $G_{i,t}$ is an unbiased estimate of $Q^\pi$ and
yet one roll-out can land far from it. That is a variance problem, not a bias
problem.

## When an estimator is unbiased

Two conditions are enough:

1. the samples come from the correct distribution;
2. the estimator is a linear function of them, built from sums, averages, and
   constant multiples.

Unbiasedness then follows from linearity of expectation alone. The sample mean
$\bar X=\frac1n\sum_i X_i$ is the canonical case:

$$
\mathbb{E}[\bar X]=\frac1n\sum_{i=1}^{n}\mathbb{E}[X_i]=\mu.
$$

Several estimators in these notes are unbiased for exactly this reason. The
reward-to-go is unbiased for $Q^\pi$ because $Q^\pi$ is *defined* as its
expectation. A state-dependent baseline leaves the policy gradient unbiased
because it factors out of the expectation over actions, as shown in
[Policy Improvement](../policy-optimization/policy-improvement.md). The
temporal-difference error built from the *true* $V^\pi$ is unbiased for
$A^\pi$, because the sampled next state enters through the single linear term
$\gamma V^\pi(s_{t+1})$; see
[Actor-Critic](../policy-optimization/actor-critic/index.md).

## A worked example: estimating a variance

Let $X_1,\dots,X_n$ be independent and identically distributed with true mean
$\mu$ and true variance $\sigma^2$. The obvious variance estimator, the mean
squared deviation from the sample mean, is biased. Write

$$
S_n^2=\frac1n\sum_{i=1}^{n}(X_i-\bar X)^2
$$

and expand the deviations around the true mean $\mu$:

$$
\begin{aligned}
\sum_{i=1}^{n}(X_i-\bar X)^2
&=\sum_{i=1}^{n}\big[(X_i-\mu)-(\bar X-\mu)\big]^2\\
&=\sum_{i=1}^{n}(X_i-\mu)^2
-2(\bar X-\mu)\underbrace{\sum_{i=1}^{n}(X_i-\mu)}_{=\,n(\bar X-\mu)}
+n(\bar X-\mu)^2\\
&=\sum_{i=1}^{n}(X_i-\mu)^2-n(\bar X-\mu)^2.
\end{aligned}
$$

Taking expectations, the first term is $n\sigma^2$ and the second is
$n\,\mathrm{Var}(\bar X)=n\cdot\sigma^2/n=\sigma^2$:

$$
\mathbb{E}\left[\sum_{i=1}^{n}(X_i-\bar X)^2\right]
=n\sigma^2-\sigma^2=(n-1)\sigma^2,
\qquad
\mathbb{E}[S_n^2]=\frac{n-1}{n}\sigma^2.
$$

The bias is $-\sigma^2/n$: the estimate is systematically too small. The
derivation also says where it went. Exactly one $\sigma^2$ is lost, and it is
the price of using $\bar X$ in place of $\mu$.

### The correction

Since the bias is a fixed multiple of the quantity we are estimating, rescaling
removes it. Dividing by $n-1$ instead of $n$ gives an unbiased estimator:

$$
S_{n-1}^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar X)^2,
\qquad
\mathbb{E}[S_{n-1}^2]=\sigma^2.
$$

### Checking it on four outcomes

Take $n=2$ and let $X$ be a fair coin with values $\{0,1\}$, so
$\sigma^2=0.25$. All four samples are equally likely.

| Sample | $\bar X$ | $S_2^2$ | $S_1^2$ |
| --- | --- | --- | --- |
| $(0,0)$ | $0$ | $0$ | $0$ |
| $(0,1)$ | $0.5$ | $0.25$ | $0.5$ |
| $(1,0)$ | $0.5$ | $0.25$ | $0.5$ |
| $(1,1)$ | $1$ | $0$ | $0$ |

Averaging the columns gives $\mathbb{E}[S_2^2]=0.125$, half the true variance,
and $\mathbb{E}[S_1^2]=0.25=\sigma^2$. No single row of the last column equals
$0.25$, which is the earlier point restated: unbiasedness is a property of the
average, not of any one estimate.

### Two ways to see the $n-1$

**Degrees of freedom.** The $n$ deviations $X_i-\bar X$ are not free. They
satisfy one linear constraint, $\sum_i (X_i-\bar X)=0$, so the last one is
determined by the first $n-1$. Only $n-1$ independent pieces of information are
present, and dividing by $n$ claims more than that.

**The mean is the culprit, not the variance.** If $\mu$ happens to be known,

$$
\frac1n\sum_{i=1}^{n}(X_i-\mu)^2
$$

is already unbiased, with $n$ in the denominator. The bias appears only because
$\mu$ has to be estimated from the same data that the deviations are measured
against. A center chosen to sit close to the sample makes the deviations look
smaller than they are.

The same conclusion follows from a form that never mentions $\bar X$. For any
$i\neq j$ we have $\mathbb{E}[(X_i-X_j)^2]=2\sigma^2$, so

$$
\frac{1}{n(n-1)}\sum_{i<j}(X_i-X_j)^2
$$

is unbiased by construction. It is identically equal to $S_{n-1}^2$, which
explains why the factor $n-1$ is the natural one.

## Two ways bias appears

The variance example is a template for both mechanisms that create bias in RL.

### A nonlinear function of an unbiased estimate

Linearity of expectation is what made the sample mean unbiased. Apply a
nonlinear function and it generally breaks. To estimate $\mu^2$ we might square
the sample mean:

$$
\mathbb{E}[\bar X^2]
=\mathrm{Var}(\bar X)+\left(\mathbb{E}[\bar X]\right)^2
=\mu^2+\frac{\sigma^2}{n}\neq\mu^2 .
$$

The bias is positive for every $n$, even though $\bar X$ itself is unbiased for
$\mu$. This is an instance of Jensen's inequality, and it is the same mechanism
behind the overestimation in $Q$-learning: with
$\mathbb{E}[\max_a \hat Q(s,a)]\ge\max_a \mathbb{E}[\hat Q(s,a)]$, an unbiased
$\hat Q$ still yields a maximum that is too large.

!!! warning "Unbiasedness does not survive transformation"
    $S_{n-1}^2$ is unbiased for $\sigma^2$, but $S_{n-1}=\sqrt{S_{n-1}^2}$ is
    biased for $\sigma$, and biased low: the square root is concave, so
    $\mathbb{E}[\sqrt{Y}]<\sqrt{\mathbb{E}[Y]}$. An unbiased estimate of a
    quantity is not an unbiased estimate of a function of that quantity.

### An approximation in place of the truth

The other mechanism is the one behind $S_n^2$: measuring against a quantity
estimated from the same data. In RL this takes two forms.

A learned critic $V_\phi^\pi$ replaces the true $V^\pi$, and its error
$\epsilon=V_\phi^\pi-V^\pi$ passes straight into the advantage estimate. When
the target is built by **bootstrapping**, $y=r+\gamma V_{\phi^-}^\pi(s')$, the
critic's own prediction becomes its label, which is the self-referential
version of the same problem.

Sampling from the wrong distribution biases an estimate as well: using data
from an older policy, drawing from a replay buffer without correction, or
correcting with importance weights and then clipping them. See
[Off-policy Learning](../policy-optimization/off-policy.md).

## Why RL cannot just divide by $n-1$

The variance example was fortunate. Its bias was proportional to the unknown
$\sigma^2$, so a constant factor cancelled it exactly. In general the bias
depends on quantities we do not know, and no fixed rescaling exists. The critic
error $\epsilon$ is exactly of that kind: if we knew it, we would already know
$V^\pi$.

RL therefore does not try to remove bias. It manages it. An $n$-step return or
generalized advantage estimation dials the amount of bootstrapping, trading
bias against variance along a continuum. Twin critics attack the specific
positive bias created by the maximum. Importance sampling corrects a known
mismatch of distributions.

## Bias is not automatically bad

What we care about is total error, and for a squared loss it splits into two
parts:

$$
\mathbb{E}\left[(\hat\theta-\theta)^2\right]
=\mathrm{bias}(\hat\theta)^2+\mathrm{Var}(\hat\theta).
$$

Accepting a little bias to remove a lot of variance often lowers this sum. For
policy gradients the requirement is weaker still: the update improves the
policy as long as the estimated direction has a positive inner product with the
true gradient. That is why actor-critic methods are willing to use a critic
they know to be wrong.

---

# 中文版本

强化学习里几乎没有能精确计算的量。回报、价值函数、策略梯度都是从有限的采样交
互中**估计**出来的。于是有两件事决定算法能不能用：估计值平均下来是否指向正确
答案，以及它围绕这个平均值散得有多开。前者是偏差，后者是方差。本页先用一个纯
统计的例子把两个定义钉死，再指出偏差在强化学习里从哪些地方进来。

## 什么是估计量

我们想知道一个未知的数 $\theta$。我们观测到随机样本 $X_1,\dots,X_n$，用某个
函数 $T$ 把它们压成一个数：

$$
\hat\theta = T(X_1,\dots,X_n).
$$

要抓住的关键是 $\hat\theta$ 本身是**随机变量**：换一批样本就得到另一个值。它
的**偏差**是它的均值与真值之差，

$$
\mathrm{bias}(\hat\theta)=\mathbb{E}[\hat\theta]-\theta,
$$

其中期望对样本分布取，$\theta$ 是常数。这个差为零时称估计量**无偏**，否则称
**有偏**。它的**方差** $\mathrm{Var}(\hat\theta)$ 描述 $\hat\theta$ 围绕自身
均值 $\mathbb{E}[\hat\theta]$ 的散布程度。

无偏说的是把实验重复无穷多次，估计值的平均恰好落在 $\theta$ 上，它对任何单次
采样不作保证。把两者分开在强化学习里很重要：reward-to-go $G_{i,t}$ 是
$Q^\pi$ 的无偏估计，但单条轨迹可能离它很远。那是方差问题，不是偏差问题。

## 什么情况下无偏

两个条件就够了：

1. 样本来自正确的分布；
2. 估计量是这些样本的线性函数，只由求和、求平均、乘常数构成。

此时无偏性仅靠期望的线性性就能得到。样本均值 $\bar X=\frac1n\sum_i X_i$ 是最
典型的例子：

$$
\mathbb{E}[\bar X]=\frac1n\sum_{i=1}^{n}\mathbb{E}[X_i]=\mu.
$$

笔记里好几个估计量无偏都是同一个道理。reward-to-go 对 $Q^\pi$ 无偏，因为
$Q^\pi$ 本来就*被定义*为它的期望。只依赖状态的基线不改变策略梯度的期望，因为
它可以从对动作的期望里提出来，推导见
[Policy Improvement](../policy-optimization/policy-improvement.md)。用**真**
$V^\pi$ 构造的时序差分误差对 $A^\pi$ 无偏，因为采样得到的下一状态只出现在
$\gamma V^\pi(s_{t+1})$ 这一个线性项里，见
[Actor-Critic](../policy-optimization/actor-critic/index.md)。

## 完整例子：估计方差

设 $X_1,\dots,X_n$ 独立同分布，真均值 $\mu$，真方差 $\sigma^2$。最直观的方差
估计量，也就是相对样本均值的平均平方偏离，是有偏的。记

$$
S_n^2=\frac1n\sum_{i=1}^{n}(X_i-\bar X)^2,
$$

把离差围绕真均值 $\mu$ 展开：

$$
\begin{aligned}
\sum_{i=1}^{n}(X_i-\bar X)^2
&=\sum_{i=1}^{n}\big[(X_i-\mu)-(\bar X-\mu)\big]^2\\
&=\sum_{i=1}^{n}(X_i-\mu)^2
-2(\bar X-\mu)\underbrace{\sum_{i=1}^{n}(X_i-\mu)}_{=\,n(\bar X-\mu)}
+n(\bar X-\mu)^2\\
&=\sum_{i=1}^{n}(X_i-\mu)^2-n(\bar X-\mu)^2.
\end{aligned}
$$

取期望，第一项是 $n\sigma^2$，第二项是
$n\,\mathrm{Var}(\bar X)=n\cdot\sigma^2/n=\sigma^2$：

$$
\mathbb{E}\left[\sum_{i=1}^{n}(X_i-\bar X)^2\right]
=n\sigma^2-\sigma^2=(n-1)\sigma^2,
\qquad
\mathbb{E}[S_n^2]=\frac{n-1}{n}\sigma^2.
$$

偏差是 $-\sigma^2/n$，即系统性偏小。这个推导还告诉我们它去哪了：恰好少了一个
$\sigma^2$，而这正是用 $\bar X$ 冒充 $\mu$ 的代价。

### 修正

既然偏差是待估量的固定倍数，重新缩放就能抵消。把分母从 $n$ 换成 $n-1$ 即得无
偏估计：

$$
S_{n-1}^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar X)^2,
\qquad
\mathbb{E}[S_{n-1}^2]=\sigma^2.
$$

### 用四种结果验证

取 $n=2$，$X$ 是取值 $\{0,1\}$ 的公平硬币，于是 $\sigma^2=0.25$。四种样本等
概率出现。

| 样本 | $\bar X$ | $S_2^2$ | $S_1^2$ |
| --- | --- | --- | --- |
| $(0,0)$ | $0$ | $0$ | $0$ |
| $(0,1)$ | $0.5$ | $0.25$ | $0.5$ |
| $(1,0)$ | $0.5$ | $0.25$ | $0.5$ |
| $(1,1)$ | $1$ | $0$ | $0$ |

对每列求平均得 $\mathbb{E}[S_2^2]=0.125$，恰是真方差的一半，而
$\mathbb{E}[S_1^2]=0.25=\sigma^2$。注意最后一列没有任何一行等于 $0.25$，这把
前面那句话又说了一遍：无偏是平均值的性质，不是单个估计值的性质。

### 从两个角度理解 $n-1$

**自由度。** $n$ 个离差 $X_i-\bar X$ 并不自由，它们满足一条线性约束
$\sum_i (X_i-\bar X)=0$，最后一个由前 $n-1$ 个决定。独立的信息只有 $n-1$
份，除以 $n$ 就是把信息量算多了。

**问题出在均值，不在方差。** 如果 $\mu$ 恰好已知，

$$
\frac1n\sum_{i=1}^{n}(X_i-\mu)^2
$$

本身就是无偏的，分母就是 $n$。偏差之所以出现，只是因为 $\mu$ 必须从同一批数
据估出来，而离差又是相对它度量的。一个被数据拽到样本附近的中心，会让离差看起
来比真实的小。

换一个完全不提 $\bar X$ 的写法能得到同样结论。对任意 $i\neq j$ 有
$\mathbb{E}[(X_i-X_j)^2]=2\sigma^2$，所以

$$
\frac{1}{n(n-1)}\sum_{i<j}(X_i-X_j)^2
$$

由构造即无偏。它与 $S_{n-1}^2$ 恒等，这也解释了为什么 $n-1$ 是自然的系数。

## 偏差的两种来源

方差这个例子同时是强化学习中两种偏差机制的模板。

### 对无偏估计做非线性变换

样本均值无偏靠的是期望的线性性。一旦套上非线性函数，通常就不成立。想估
$\mu^2$，我们可能会把样本均值平方：

$$
\mathbb{E}[\bar X^2]
=\mathrm{Var}(\bar X)+\left(\mathbb{E}[\bar X]\right)^2
=\mu^2+\frac{\sigma^2}{n}\neq\mu^2 .
$$

对任意 $n$ 偏差都为正，尽管 $\bar X$ 对 $\mu$ 是无偏的。这是 Jensen 不等式的
一个实例，也正是 $Q$-learning 过估计背后的机制：由
$\mathbb{E}[\max_a \hat Q(s,a)]\ge\max_a \mathbb{E}[\hat Q(s,a)]$，即使
$\hat Q$ 无偏，取最大值之后仍然偏大。

!!! warning "无偏性不能穿过变换传递"
    $S_{n-1}^2$ 对 $\sigma^2$ 无偏，但 $S_{n-1}=\sqrt{S_{n-1}^2}$ 对
    $\sigma$ 有偏，而且偏小：开方是凹函数，所以
    $\mathbb{E}[\sqrt{Y}]<\sqrt{\mathbb{E}[Y]}$。某个量的无偏估计，不是这个
    量的函数的无偏估计。

### 用近似替代真值

另一种机制就是 $S_n^2$ 的毛病：拿同一批数据估出来的量当基准去度量。它在强化
学习里有两副面孔。

学出来的 critic $V_\phi^\pi$ 替代了真 $V^\pi$，它的误差
$\epsilon=V_\phi^\pi-V^\pi$ 直接传进优势估计。当目标是靠**bootstrapping**
构造的，即 $y=r+\gamma V_{\phi^-}^\pi(s')$，critic 自己的预测成了自己的标签，
这是同一个问题的自指版本。

从错误的分布采样同样带来偏差：使用旧策略的数据、不做修正就从 replay buffer
采样、或者做了重要性采样又把权重截断。见
[Off-policy Learning](../policy-optimization/off-policy.md)。

## 为什么强化学习里不能照样除以 $n-1$

方差那个例子是运气好。它的偏差正比于未知的 $\sigma^2$，因此一个常数因子就能
精确抵消。一般情况下偏差依赖我们不知道的量，不存在这样的固定缩放。critic 误
差 $\epsilon$ 正是这一类：如果我们知道它，就等于已经知道 $V^\pi$ 了。

所以强化学习不追求消除偏差，而是管理它。$n$-step 回报和广义优势估计通过一个
参数调节 bootstrapping 的比例，在偏差和方差之间连续取舍。双 critic 针对取最
大值造成的那个特定正偏。重要性采样纠正已知的分布不匹配。

## 有偏不等于坏

我们真正在意的是总误差，而在平方损失下它可以拆成两部分：

$$
\mathbb{E}\left[(\hat\theta-\theta)^2\right]
=\mathrm{bias}(\hat\theta)^2+\mathrm{Var}(\hat\theta).
$$

用一点偏差换掉大量方差，往往能让这个和变小。对策略梯度来说要求还更宽松：只要
估计出的方向与真梯度的内积为正，更新就在改善策略。这就是 actor-critic 敢用一
个明知不准的 critic 的理由。
