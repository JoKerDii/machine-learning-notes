# Introduction to RL

There are 7 topics and 4 exercises.

* Unlike with Supervised Learning, there would typically be **no labelled training dataset** associated with Reinforcement Learning tasks. 
* RL algorithms learn to pick “good" **actions** based on the **rewards** that they receive during training so as to maximize some notion of a **cumulative reward** instead of the reward for the next step and they can take “good" actions even without any intermediate rewards.

## 1. Markov Decision Process (MDP)

* A set of states $s \in S$

* A set of actions $a \in A$

* Action dependent transition probabilities $T(s,a,s') = P(s'|s,a)$, so that for each state $s$ and action $a$, 
  $$
  \sum_{s' \in S} T(s,a,s') = 1
  $$

* Reward functions $R(s,a,s')$, representing after one step, the reward for starting in state $s$, taking action $a$ and ending up in state $s'$.

MDPs satisfy the **Markov property** in that the transition probabilities and rewards depend only on the current state and action, and remain unchanged regardless of the history (i.e. past states and actions) that leads to the current state.

Let $X_i, i =1,2,...$ be a discrete Markov chain with states $\{s_j, j \in \N\}$. 

* For $n \geq 3$, $P[X_n = x_n | X_{n-1} = x_{n-1}, ...,X_1 = x_1] = P[X_n = x_n | X_{n-1}=x_{n-1}]$
* For $n \geq 3$, and $n-j > 1$, $P[X_n = x_n | X_{n-j} = x_{n-j}, ...,X_1 = x_1] = P[X_n = x_n | X_{n-j}=x_{n-j}]$

Therefore, the **state transition probability** for Markov State is given by
$$
P_{ss'}=\mathbb{P}[S_{t+1}=s' | S_t = s])
$$
And the **state transition probability matrix** is given by
$$
P = \begin{bmatrix} p_{11} & p_{12} &...& p_{1n}\\p_{21} & p_{22} & ... & p_{2n}\\  & ... & ... & \\p_{n1} & p_{n2} & ... & p_{nn}\end{bmatrix}
$$
Note that the sum of each row is $1$
$$
\sum_{j = 1}^n p_{ij} = 1 \quad \text{ for i = 1,..., n}
$$

> #### Exercise 40
>
> The agent takes actions $a_1, a_2, ...a_n$ starting from state $s_0$ and as a result visits states $s_1, s_2, ...s_n = s$ in that order. Given that $s_n = s$, the agent ends up at the current, what do the rewards after the $n$th step depend on?
>
> > **Answer**: Reward collected after the $n$th step do not depend on the previous states or actions, however, they depend on the current state $s$ and the current action $a_{n+1}$.

Assume that the **transition probabilities** for all the states are given as a cube $M$, whose $(i,j,k)$th entry is $M[i],[j],[k] = T(s_i, a_j, s_k) = P(s_k| s_i, a_j)$, which represents the transition probability of ending up at state $s_k$ when action $a_j$ is taken from the state $s_i$. The number of entries in $M$ is $8 \times 8 \times 4 = 256$, since there are 8 states and 4 actions. Note that for any given state, action pair $(s, a)$, the following must hold
$$
\sum _{s'} P(s'|s, a) = 1
$$

## 2. Utility Function

Utility function is a criterion used to cumulate rewards so that we can maximize its expectation in terms of **accumulated** rewards.

* **Finite horizon based utility**
  $$
  U[s_0,s_1,\ldots , s_{n+m}]=U[s_0,s_1,\ldots , s_ n]\quad \text {for any positive integer } m.\\
  U[s_0,s_1,\ldots , s_ n]= \sum _{i=0}^{n} R(s_ i)\quad \text {for some fixed number of steps } n.
  $$

  * The action at state $s$ that maximizes a finite horizon based utility can depend on how many steps have been taken. For example, if we are at state s at time N, the agent will want to act greedily and take the action that leads to the immediate highest reward. However, if we are at time 0, the agent can allow to move towards areas with higher rewards while getting an immediate lower reward.

* **(Infinite horizon) discounted reward based utility**

  The goal is to continue acting (without an end) while maximizing the expected discounted reward. The discounting allows us to focus on near term rewards, and control this focus by changing $γ$.
  $$
  \begin{aligned}
  U[s_0,s_1,\ldots ] &= R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + ... \\ &= \sum _{k=0}^{\infty } \gamma ^ k R(s_ k)
  \end{aligned}
  $$

  * **Discount Factor ($\gamma$)** determines how much **importance** is to be given to the **immediate reward and future rewards**. This basically helps us to avoid infinity as a reward in continuous tasks.

    * Intuitively, if $\gamma \rightarrow 1$ we are interested in the future rewards and we will make an effort to go to end. For $\gamma = 0$, maximizing for discounted reward boils down to greedily maximizing for the immediate reward.

  * Discount reward is guaranteed to be **finite** when the maximum reward is finite. When the maximum reward is finite, the bound of discount reward is 
    $$
    \sum _{k=0}^{\infty } \gamma ^ k R(s_ k) \leq R_{max}(s) \sum^{\infty}_{k=0}\gamma^k = \frac{R_{max}(s)}{1-\gamma}
    $$

  * The action at state $s$ that maximizes a discount reward based utility **does not** depend on how many steps have been taken. Discounted reward based utility under a **markovian** setting would lead to an optimal policy that only depends on the state and is independent of the step where the state occurs.

## 3. Policy and Value Functions

Given an MDP, and a utility function $U[s_0, s_1, ..., s_n]$, our goal is to find an **optimal policy function** that maximizes the expectation of the utility. A **policy** is a function $\pi : S\to A$ (a mapping from a state to an action (or a probability distribution over the set of actions)) that assigns an action $\pi(s)$ to any state $s$, and the optimal policy is denoted as $\pi^*$. 

Key Points:

* The goal of the optimal policy function is to maximize the **expected discounted reward**, not immediate reward. 
* The optimal policy is highly dependent on the **structure of rewards.**
* Under our current **markovian** setting, it does not matter what sequence of states where visited in the past to determine what the policy should be from a given state.

### Bellman Equation:

$$
\begin{aligned}
V^*(s) &= \max _ a Q^*(s, a)\\
Q^*(s, a) &= \sum _{s'} T(s, a, s') ( R(s, a, s') + \gamma V^*(s') )
\end{aligned}
$$

where

* the **value function** $V^*(s)$ is the expected reward from starting at state $s$ and acting optimally
* the **Q-function** $Q^*(s,a)$ is the expected reward from starting at state $s$, then acting with action $a$, and acting optimally.

> #### Exercise 41
>
> Let there be 4 possible actions $a_1, a_2, a_3, a_4$ from a given state $s$, and let $Q^*$ values be as follows
> $$
> Q^*(s, a_1) = 10\\
> Q^*(s, a_2) = -1\\
> Q^*(s, a_3) = 0\\
> Q^*(s, a_4) = 11
> $$
> Let $s'$ be a state that can be reached from $s$ by taking the action $a_1$. Let
> $$
> \begin{aligned}
> T(s, a_1, s') &= 1\\
> R(s, a_1, s') &= 5\\
> \gamma &= 0.5
> \end{aligned}
> $$
> What is the value of $V^*(s)$ and $V^*(s')$?
>
> > **Answer**: $V^*(s) = 11, V^*(s') = 10$.
>
> > **Solution**:
> > $$
> > V^*(s) = \max _ a Q^*(s, a)= \max(10,-1,0,11) = 11
> > $$
> > Note that $T$ denotes probabilities, so the following must be true
> > $$
> > \sum _{s'} T(s, a, s') = 1
> > $$
> > since $T(s, a_1, s') = 1$, we can reduce the Bellman Equation 
> > $$
> > Q^*(s, a) = \sum _{s'} T(s, a, s') ( R(s, a, s') + \gamma V^*(s') )\\
> > \implies Q^*(s, a_1) = T(s, a_1, s') ( R(s, a_1, s') + \gamma V^*(s') )
> > $$
> > Then $V^*(s')$ can be computed as
> > $$
> > 10 = 1* (5 + 0.5* V^*(s') )\\
> > V^*(s') = 5/0.5 = 10
> > $$

## 4. Value Iteration & Q-value Iteration

**Q-value iteration update rule** $Q^*(s, a)$ is defined by plugging in the first equation into the second equation of Bellman equation.
$$
Q^*(s, a) = \sum _{s'} T(s, a, s')\left(R(s, a, s') + \gamma \max _{a'} Q^*(s', a')\right)
$$
**Value iteration update rule** $V^*_{k+1}(s)$ is defined by maximizing the Q-value iteration update rule.
$$
V^*_{k+1}(s) = \max Q^*(s, a) = \max _ a\left[\sum _{s'} T(s, a, s') \left(R(s, a, s') + \gamma V^*_ k(s')\right)\right]
$$
where $V_k^*(s)$ is the expected reward from state $s$ after acting optimally for $k$ steps.

> #### Exercise 42
>
> Consider an agent is trying to navigate a one-dimensional grid consisting of 5 cells, with +1 reward in the final cell. 
>
> Let $V^*(i)$ denotes the value function of state $i$, the $i^{th}$ cell starting from left, $V_k^*(i)$ denotes the value function estimate at state $i$ at the $k^{th}$ step of the value iteration algorithm. Let $V_0^*(i)$ denote the initialization of this estimate. Use the discount factor $\gamma = 0.5$. 
>
> Write down $V_k^* = \begin{bmatrix}  V^*_ k(1)& V^*_ k(2)& V^*_ k(3)& V^*_ k(4)& V^*_ k(5)\end{bmatrix}$. How many steps it takes starting from $V_0^*$ for the value function updates to converge to the optimal value function $V^*$.
>
> > **Answer**: 
> > $$
> > \begin{aligned}
> > V^*_0 &= \begin{bmatrix} 0& 0& 0& 0& 0\end{bmatrix}\\
> > V^*_1 &= \begin{bmatrix} 0& 0& 0& 0& 1\end{bmatrix}\\
> > V^*_2 &= \begin{bmatrix} 0& 0& 0& 0.5& 1\end{bmatrix}\\
> > V^*_3 &= \begin{bmatrix} 0& 0& 0.25& 0.5& 1\end{bmatrix}\\
> > V^*_4 &= \begin{bmatrix} 0& 0.125& 0.25& 0.5& 1\end{bmatrix}\\
> > V^*_5 &= \begin{bmatrix} 0.0625& 0.125& 0.25& 0.5& 1\end{bmatrix}
> > \end{aligned}
> > $$
> > It takes 5 steps to converge.
>
> > **Solution**: 
> >
> > Since the agent takes action to reach the 5th cell, and no more action,  we set 
> > $$
> > V^*_{k+1}(5)=V^*_{k}(5)
> > $$
> > Given the situation we have the reward function $R(s, a, s')=R(s)$, $R(s=5) = 1$ and $R(s) = 0$ otherwise.
> >
> > The calculation of $V_k^*$ is below:
> >
> > For $k=1$,
> > $$
> > V_1^*(1) = 0 + \gamma V_0^*(2) = 0\\
> > V_1^*(2) = 0 + \gamma V_0^*(3) = 0\\
> > V_1^*(3) = 0 + \gamma V_0^*(4) = 0\\
> > V_1^*(4) = 0 + \gamma V_0^*(5) = 0\\
> > V_1^*(5) = V_0^*(5) = 1\\
> > $$
> > For $k=2$,
> > $$
> > \begin{aligned}
> > V_2^*(1) &= 0 + \gamma V_1^*(2) = 0\\
> > V_2^*(2) &= 0 + \gamma V_1^*(3) = 0\\
> > V_2^*(3) &= 0 + \gamma V_1^*(4) = 0\\
> > V_2^*(4) &= 0 + \gamma V_1^*(5) = 0.5\\
> > \end{aligned}
> > $$
> > For $k=3$,
> > $$
> > \begin{aligned}
> > V_3^*(1) &= 0 + \gamma V_2^*(2) = 0\\
> > V_3^*(2) &= 0 + \gamma V_2^*(3) = 0\\
> > V_3^*(3) &= 0 + \gamma V_2^*(4) = 0.25\\
> > V_3^*(4) &= 0 + \gamma V_2^*(5) = 0.5\\
> > \end{aligned}
> > $$
> > For $k=4$,
> > $$
> > \begin{aligned}
> > V_4^*(1) &= 0 + \gamma V_3^*(2) = 0\\
> > V_4^*(2) &= 0 + \gamma V_3^*(3) = 0.125\\
> > V_4^*(3) &= 0 + \gamma V_3^*(4) = 0.25\\
> > V_4^*(4) &= 0 + \gamma V_3^*(5) = 0.5\\
> > \end{aligned}
> > $$
> > For $k = 5$,
> > $$
> > \begin{aligned}
> > V_5^*(1) &= 0 + \gamma V_4^*(2) = 0.0625\\
> > V_5^*(2) &= 0 + \gamma V_4^*(3) = 0.125\\
> > V_5^*(3) &= 0 + \gamma V_4^*(4) = 0.25\\
> > V_5^*(4) &= 0 + \gamma V_4^*(5) = 0.5\\
> > \end{aligned}
> > $$
> > After the $5$th step, the reward from the rightmost cell in the grid gets propagated to the leftmost state after which the value function estimate $V^∗_k$ stops updating. 

> #### Exercise 43
>
> Let the number of states and actions be $|S|$ and $|A|$, respectively, What is the complexity of an iteration of the value algorithm?
>
> > **Answer**: $O(|S|^2 \cdot |A|)$
>
> > **Solution**: 
> >
> > We update the expected reward for each state in every iteration - there are $|S|$ states. For each state, we investigate a maximum of $|A|$ possible actions and for each such action there are $|S|$ possible transitions at the most. 

## 5. Estimating Inputs

In MDP setting for RL algorithms, we have a tuple $<S,A,T,R>$, where $S$ (state space) and $A$ (action space) are unknown, while $T$ (transition probabilities) and $R$ (reward structure) should be estimated from $S$ and $A$ in the non-deterministic noisy real world scenarios:
$$
\hat{T} = \frac{\text {count}(s, a, s')}{\displaystyle \sum _{s'} \text {count}(s, a, s')}\\
\hat{R} = \frac{\displaystyle \sum _{t=1}^{\text {count}(s, a, s')} R_ t(s, a, s')}{\text {count}(s, a, s')}
$$
However, the issue with this estimation is that statistics for $\hat{T}$ and $\hat{R}$ can be collected for a given state only when the agent visits the state during the estimation process, so most likely the agent only circle around a subset of state space and won't collect much statistics for estimation. Another issue is that if a certain state is visited less, the estimation for this state would be very noisy.

#### Model Free vs. Model Based Approaches

Suppose we have $K$ samples and we are estimating the expectation of a function $f(x)$:
$$
\mathbb{E}[f(X)] = \sum_x p(x) \cdot f(x)
$$

* Model based approach works by sampling $K$ points from the distribution $P(x)$ and estimating the probability distribution
  $$
  \hat{p}(X) = \text {count}(x)/K
  $$
  before estimating the expectation of $f(X)$ as follows:
  $$
  \mathbb{E}[f(X)] \approx \displaystyle \sum \hat{p}(x) f(x)
  $$

* Model free approach would sample $K$ points from $P(X)$ and directly estimate the expectation of $f(X)$ as follows:
  $$
  \mathbb{E}[f(X)] \approx \frac{\displaystyle \sum _{i=1}^ K f(X_ i)}{K}
  $$

## 6. Q-value Iteration by Sampling

We want to find optimal $Q^*$ function by sampling for tasks where we don't have access to the exact $T,R$ functions. Suppose the agent starts out from state $s_1$ and collect a few samples. For example the reward is
$$
\text{Sample 1:} \quad R(s,a,s_1') + \gamma \max_{a'}Q(s'_1, a')\\
\text{Sample k:} \quad R(s,a,s_k') + \gamma \max_{a'}Q(s'_k, a')
$$
Let $S_ k^{Q(s, a)}$ denotes the $k^{th}$ sample of $Q(s, a)(k = i+1)$. The Q-function can be the average reward
$$
Q(s,a)=\frac{1}{k} \sum^k_{i=1}S_ i^{Q(s, a)} = \frac{1}{k}\sum^k_{i=1}(R(s,a,s_i') + \gamma \max_{a'}Q(s_i', a'))
$$
However, we want to get Q-function updated each time we get new sample incrementally, rather than waiting for collecting all samples. Therefore, we apply **exponential running average**, which gives more weight to current samples compared to previous samples.

The exponential running average is 
$$
\bar{x}_n = \frac{x_n + (1-\alpha)x_{n-1} + (1-\alpha)^2x_{n-2} + ...}{1 + (1-\alpha) + (1-\alpha)^2+ ...} = \alpha x_n + (1-\alpha)\bar{x}_{n-1}
$$
Let $S_ k^{Q(s, a)}$ denotes $k^{th}$ sample $Q(s, a)(k = i+1)$, then the exponential running average is
$$
\hat{Q}_{i+1}(s, a) = \alpha S_ k^{Q(s,a)} + (1 - \alpha )* \hat{Q}_ i(s, a)
$$
So the estimated Q-function is 
$$
Q_{i+1}(s,a) = \alpha \cdot S_ k^{Q(s, a)} + (1-\alpha) \cdot Q_i (s,a)\\
\text{where the reward} \quad S_ k^{Q(s, a)} = R(s,a,s') + \gamma \max_{a'}Q(s', a')
$$
So the steps are 

1. Initialization: $\hat{Q}_0(s, a) = 0 \quad \text { for all }\,  s, a$

2. Iterate until converge

   2a. collect samples, each can be described by the tuple $(s, a, s', R(s, a, s'))$.

   2b. compute 
   $$
   \begin{aligned}
   Q_{i+1}(s,a) &= \alpha \cdot (R(s,a,s') + \gamma \max_{a'}Q_i(s', a')) + (1-\alpha) \cdot Q_i (s,a)\\
    &= Q_i (s,a) + \alpha\cdot( R(s,a,s') + \gamma \max_{a'}Q_i(s', a') - Q_i (s,a))
    \end{aligned}
   $$

## 7. Exploration vs Exploitation

**Exploration**: to try out random actions and visit unknown states.

**Exploitation**: to take optimal action with knowledge about the environment. Specifically, to take an action $a$ from state $s$ such that current estimate of Q function  $\hat{Q}(s,a)$ is maximized.

**$\epsilon$-greedy approach**: to balance exploration and exploitation by randomly sampling an action with probability $\epsilon$ and by choosing the best currently available option with probability $1-\epsilon$.  

* Higher the $\epsilon$, higher are the chances that the agent takes a random action during the learning phase and higher are the chances that it explores new states and actions.
* As the agent learns to act well, and has sufficiently explored its environment, $ϵ$ should be decayed off so that the value and $Q$ function samples get less noisy with some of the randomness in the agent's policy eliminated.

# Additional Readings

RL-Part1

https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da

RL-Part2

https://towardsdatascience.com/reinforcement-learning-markov-decision-process-part-2-96837c936ec3