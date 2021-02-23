# Linear Classifier and Perceptron

There are 3 topics and 3 exercises.

> #### **Basic Concepts:**
>
> * Feature vectors, labels: $x \in R^d, y \in \{ -1, 1\}$.
> * Training set: $S_n = \{(x^{(i)}, y^{(i)}), i=1,...,n\}$
> * Classifier: $h: R^d \to \{-1, 1\}, X^+ = \{x \in R^d: h(x) = 1\}, X^- = \{x \in R^d: h(x) = -1\}$
> * Training error / test error: $\epsilon_n(h) = \frac{1}{n} \sum^n_{i=1}[[h(x^{(i)}) \neq y^{(i)}]]$
> * Set of classifier: $h \in H$, where $H$ is a set of classifiers.

## 1. Linear Classifier

**Classifiers** are **mappings** that take **feature vectors as input** and produce **labels as output**. **Hypothesis space** is the set of possible classifiers.

Given $\theta$ and $\theta_0$, a **linear classifier** $h:X→{−1,0,+1}$ is a function that outputs +1 if $\theta⋅x+\theta_0$ is positive, 0 if it is zero, and −1 if it is negative. In other words, $h(x)=sign(\theta x+\theta_0)$ or the set of linear classifiers: $\{h(x; \theta, \theta_0)=sign(\theta x+\theta_0), \theta \in R^d, \theta_0 \in R\}$.

**Linear separation**: Training examples $S_n = \{(x^{(i)}, y^{(i)}), i = 1, ..., n\}$ are linearly separable if there exists a parameter vector $\hat{\theta}$ and offset parameter $\hat{\theta_0}$ such that $y^{(i)}(\hat{\theta}x^{(i)} + \hat{\theta_0})>0$ for all $i = 1, ..., n$.

> #### **Exercise 11:**
>
> For the $i$th training data $(x_i,y_i)$, what values can $y^{(i)}$ take, **conventionally** (in the context of linear classifiers)? what values can $sign(\theta⋅x^{(i)})$ take? Choose all those apply.
>
> A. -1
>
> B. +1
>
> C. 0
>
> D. +10
>
> > **Answer**: AB ; ABC
>
> > **Solution**: By the convention of linear classification, because $y^{(i)}$ is a label, it can take −1 or +1. Note that 0 is not a possible value.

## 2. The Perceptron Algorithm with 0-1 Loss

The perceptron algorithm without offset: 

$$
\text{Perceptron }(\{(x^{(i)},y^{(i)}),i=1,...,n\},T):\\
\text{initialize } \theta=0\text{(vector)}; ~~~~~~~~~~~~~~~~~\\
\text{for } t=1,...,T \text{ do}~~~~~~~~~~~~~~~~~\\
\text{for } i=1,...,n \text{ do}~~~~~~~~~~\\
~~~~~~~\text{if } y^{(i)}(\theta⋅x^{(i)})≤0 \text{ then}\\
~~~~~~~~~~~~~~\text{update } \theta=\theta+y^{(i)}x^{(i)}\\
$$


The perceptron algorithm with offset: 
$$
\text{Perceptron }(\{(x^{(i)},y^{(i)}),i=1,...,n\},T):\\
\text{initialize } \theta=0\text{(vector)}; \theta_0 = 0 \text{(scalar)}\\
\text{for } t=1,...,T \text{ do}~~~~~~~~~~~~~~~~~\\
\text{for } i=1,...,n \text{ do}~~~~~~~~~~\\
~~~~~~~\text{if } y^{(i)}(\theta⋅x^{(i)})≤0 \text{ then}\\
~~~~~~~~~~~~~~\text{update } \theta=\theta+y^{(i)}x^{(i)}\\
~~~~~~~~~~~~\text{update } \theta_0=\theta_0+y^{(i)}\\
$$

Average Perceptron Algorithm:

* In this case, all contributions are averaged out. This is to prevent a big jump from old $\theta$ to new $\theta$ in each iteration. The returned parameters $\theta$  are an average of the $\theta$s across the $nT$ steps:

$$
θ_{final}={1\over nT}(θ^{(1)}+θ^{(2)}+...+θ^{(nT)})
$$

* The problem is that by going through the data in one iteration, updates made to $\theta$ s can cause the data that was processed earlier to be misclassified. So going through multiple iterations could correct this.



#### A **geometrical description** of updating decision boundary:

Let's initiate $\theta = 0$, assume there is no offset, and we have two points - one with positive label and one with negative label. For the 1st point with $y^1 = 1$, if $y^1(\theta x^1) < 0$, then $\theta = \theta + y^1 x^1 = x^1$. For the 2nd point with $y^1 = -1$, if $y^2(\theta x^2) < 0$, then $\theta = \theta + y^2 x^2 = x^1 - x^2$.

This is desirable since $\theta$ is the normal vector of the decision boundary, when $\theta = x^1 - x^2$, the decision boundary lies perpendicular to the vector $x^1 - x^2$, which is a vector from negative point to positive point. After updating $\theta$, the decision boundary well splits the data.



> #### Exercise 12:
>
> When a mistake is spotted by perceptron (i.e. $y^{(i)}(\theta \cdot x^{(i)} + \theta _0) \leq 0$), do the updated values of $\theta$ and $\theta_0$ provide a better prediction? In other words, is $y^{(i)}((\theta +y^{(i)} x^{(i)}) \cdot x^{(i)} + \theta _0 + y^{(i)})$ always greater than or equal to $y^{(i)}(\theta \cdot x^{(i)} + \theta _0)$?
>
> A. Yes, because $\theta+y^{(i)}x^{(i)}$ is always larger than $\theta$
>
> B. Yes, because $(y^{(i)})^2\| x^{(i)}\| ^2 + (y^{(i)})^2 \geq 0$.
>
> C. No, because $(y^{(i)})^2\| x^{(i)}\| ^2 - (y^{i})^2 \leq 0$
>
> D.No, because $\theta + y^{(i)} x^{(i)}$ is always larger than $\theta$
>
> > **Answer**: B
>
> > **Solution**: Compare the two quantities by subtraction:
> > $$
> > y^{(i)}((\theta +y^{(i)} x^{(i)}) \cdot x^{(i)} + \theta _0 + y^{(i)}) - y^{(i)}(\theta \cdot x^{(i)} + \theta _0) = (y^{(i)})^2 \| x^{(i)}\| ^2 + (y^{(i)})^2 = (y^{(i)})^2(\| x^{(i)}\| ^2 + 1)) > 0
> > $$
> > This is desirable, since the update of $\theta$ always decrease the training error.



## 3. Perceptron Training Error

For a given example $i$, we defined the training error as 1 if  $y^{(i)}(\theta \cdot x^{(i)} + \theta _0) \leq 0$, and 0 otherwise:
$$
\varepsilon _ i(\theta , \theta _0) = \big [\big [ y^{(i)}(\theta \cdot x^{(i)} + \theta _0) \leq 0 \big ]\big ]
$$

> #### **Exercise 13:** 
>
> We have a linear classifier given by $\theta, \theta_0$. After the perceptron update using example $i$, the training error $\epsilon_i(\theta, \theta+0)$ for that example can (select all those apply):
>
> A. Increase
>
> B. Stay the same 
>
> C. Decrease
>
> > **Answer**: BC
>
> > **Solution**: $\big [\big [ y^{(i)}(\theta \cdot x + \theta _0) \leq 0 \big ]\big ]$ becomes 0 or stays 1.



## 4. *Pegasos Algorithm

Please refer to [original paper](https://link.springer.com/article/10.1007/s10107-010-0420-4) for more information.

The update rule of Pegasos Algorithm without offset:
$$
\text{Pegasos update rule } \left(x^{(i)}, y^{(i)}, \lambda , \eta , \theta \right):\\
\text{if } y^{(i)}(\theta \cdot x^{(i)}) \leq 1 \text{ then}~~~~~~~~~~~~~~~~~~~\\
~~~~\text{update } \theta = (1 - \eta \lambda ) \theta + \eta y^{(i)}x^{(i)}\\
\text{else: }~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\\
\text{update } \theta = (1 - \eta \lambda ) \theta~~~~~~~~~~~~~~\\
$$

The update rule of Pegasos Algorithm with offset:
$$
\text{Pegasos update rule } \left(x^{(i)}, y^{(i)}, \lambda , \eta , \theta \right):\\
\text{if } y^{(i)}(\theta \cdot x^{(i)}) \leq 1 \text{ then}~~~~~~~~~~~~~~~~~~~\\
~~~~\text{update } \theta = (1 - \eta \lambda ) \theta + \eta y^{(i)}x^{(i)}\\
~~~~\text{update } \theta_0 = \theta_0 + \eta y^{(i)}~~~~~~~~~~~~~~~~\\
\text{else: }~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\\
\text{update } \theta = (1 - \eta \lambda ) \theta~~~~~~~~~~~~~~\\
$$

where The $\eta$ parameter is a decaying factor that will decrease over time. The $\lambda$ parameter is a regularizing parameter. Note that the magnitude of $\theta_0$ should not be penalized.

#### Good ideas in Pegasos Algorithm:

* Regularization
* Hinge loss
* Subgradient updates
* Decaying learning rate

