# Introduction to RL

There are topics and exercises.

* Unlike with Supervised Learning, there would typically be **no labelled training dataset** associated with Reinforcement Learning tasks. 
* RL algorithms learn to pick “good" **actions** based on the **rewards** that they receive during training so as to maximize some notion of a **cumulative reward** instead of the reward for the next step and they can take “good" actions even without any intermediate rewards.

## Markov Decision Process (MDP)

* A set of states $s \in S$

* A set of actions $a \in A$

* Action dependent transition probabilities $T(s,a,s') = P(s'|s,a)$, so that for each state $s$ and action $a$, 
  $$
  \sum_{s' \in S} T(s,a,s') = 1
  $$

* Reward functions $R(s,a,s')$, representing after one step, the reward for starting in state $s$, taking action $a$ and ending up in state $s'$.

MDPs satisfy the **Markov property** in that the transition probabilities and rewards depend only on the current state and action, and remain unchanged regardless of the history (i.e. past states and actions) that leads to the current state.

Let $X_i, i =1,2,...$ be a discrete Markov chain with states $\{s_j, j \in \N\}$. 

* For $n \geq 3$, $P[X_n = x_n | X_{n-1} = x_{n-1}, X_1 = x_1] = P[X_n = x_n | X_{n-1}=x_{n-1}]$
* For $n \geq 3$, and $n-j > 1$, $P[X_n = x_n | X_{n-j} = x_{n-j}, X_1 = x_1] = P[X_n = x_n | X_{n-j}=x_{n-j}]$

> #### Exercise 40
>
> The agent takes actions $a_1, a_2, ...a_n$ starting from state $s_0$ and as a result visits states $s_1, s_2, ...s_n = s$ in that order. Given that $s_n = s$, the agent ends up at the current, what do the rewards after the $n$th step depend on?
>
> > **Answer**: Reward collected after the $n$th step do not depend on the previous states or actions, however, they depend on the current state $s$ and the current action $a_{n+1}$.

Assume that the **transition probabilities** for all the states are given as a table $M$, whose $(i,j,k)$th  entry is $M[i],[j],[k] = T(s_i, a_j, s_k) = P(s_k| s_i, a_j)$, which represents the transition probability of ending up at state $s_k$ when action $a_j$ is taken from the state $s_i$. The number of entries in $M$ is $8 \times 8 \times 4 = 256$, since there are 8 states and 4 actions. Note that for any given state, action pair $(s, a)$, the following must hold
$$
\sum _{s'} P(s'|s, a) = 1
$$

## Utility Function

## Additional Readings

https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da

