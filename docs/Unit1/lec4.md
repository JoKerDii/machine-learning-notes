# Linear Classification and Generalization

There are 3 topics and 2 exercises

## 1. Gradient Descent

Assume $\theta \in R$. Our goal is to find $\theta$ that minimizes
$$
J(\theta , \theta _0) = \frac{1}{n} \sum _{i=1}^{n} \text {Loss}_ h (y^{(i)} (\theta \cdot x^{(i)} + \theta _0 )) + \frac{\lambda }{2} \mid \mid \theta \mid \mid ^2
$$
through **gradient descent**. In other words, we will

1. Start $\theta$ at an arbitrary location: $\theta \leftarrow \theta _{start}$.
2. Update $\theta$ repeatedly with $\theta \leftarrow \theta - \eta \frac{\partial J(\theta , \theta _0)}{\partial \theta }$ until $\theta$ does not change significantly.

As $\theta$ is an $n$-dimensional vector, the updating rule is written as $\theta \leftarrow \theta - \eta \nabla J(\theta)$, where $\nabla J(\theta) = \begin{bmatrix}  \frac{\partial J(\theta , \theta _0)}{\partial \theta_1 }\\ \frac{\partial J(\theta , \theta _0)}{\partial \theta_2 } \\ \vdots \\ \frac{\partial J(\theta , \theta _0)}{\partial \theta_d } \end{bmatrix}$.

## 2. Stochastic Gradient Descent (SGD)

The objective is,
$$
J(\theta , \theta _0) = \frac{1}{n} \sum _{i=1}^{n} \text {Loss}_ h (y^{(i)} (\theta \cdot x^{(i)} + \theta _0 )) + \frac{\lambda }{2} \mid \mid \theta \mid \mid ^2 = \frac{1}{n} \sum _{i=1}^{n}\big [ \text {Loss}_ h (y^{(i)} (\theta \cdot x^{(i)} + \theta _0 )) + \frac{\lambda }{2} \mid \mid \theta \mid \mid ^2 \big ]
$$
where $J_i(\theta, \theta_0) = \text {Loss}_ h (y^{(i)} (\theta \cdot x^{(i)} + \theta _0 )) + \frac{\lambda }{2} \mid \mid \theta \mid \mid ^2 $ are from some random sample in SGD.

In SGD, we choose $i \in \big \{ 1,...,n \big \}$ at random and update $\theta$ such that
$$
\theta \leftarrow \theta - \eta \nabla _{\theta } \big [\text {Loss}_ h(y^{(i)}(\theta \cdot x^{(i)} + \theta _0) ) + \frac{\lambda }{2}\mid \mid \theta \mid \mid ^2 \big ]
$$

> #### **Exercise 15:**
>
> What is $\nabla _{\theta }\big [ \text {Loss}_ h(y^{(i)}(\theta \cdot x^{(i)} + \theta _0) ) \big ]$ if $\text {Loss}_ h(y^{(i)}(\theta \cdot x^{(i)} + \theta _0))>0$?
>
> > **Answer**: $-y^{(i)}x^{(i)}$
>
> > **Solution**: If $\text {Loss}_ h(y^{(i)}(\theta \cdot x^{(i)} + \theta _0))>0$, $\text {Loss}_ h(y^{(i)}(\theta \cdot x^{(i)} + \theta _0)) = 1- y^{(i)}(\theta \cdot x^{(i)} + \theta _0)$. Thus, 
> > $$
> > \nabla _{\theta } \text {Loss}_ h(y^{(i)}(\theta \cdot x^{(i)} + \theta _0)) = -y^{(i)}x^{(i)}
> > $$
> > 

> #### **Exercise 16:** 
>
> For SGD, which of the following is true?
>
> A. As in perceptron, $\theta$ is not updated when there is no mistake
>
> B. Differently from perceptron, $\theta$ is updated even when there is no mistake
>
> > **Answer**: B
>
> > **Solution**: 
> >
> > The update step of SGD:
> > $$
> > \theta \leftarrow \theta - \eta \nabla _{\theta } \big [\text {Loss}_ h(y^{(i)}(\theta \cdot x^{(i)} + \theta _0) ) + \frac{\lambda }{2}\mid \mid \theta \mid \mid ^2 \big ]
> > $$
> > which can also be written as
> > $$
> > \theta \leftarrow \begin{cases}  (1-\lambda \eta ) \theta \text { if Loss=0} \\ (1-\lambda \eta ) \theta + \eta y^{(i)} x^{(i)} \text { if Loss>0} \end{cases}
> > $$
> > that $\theta$ is updated even when the sum of losses is 0. This is different from perceptron.

## 3. The Realizable Case - Quadratic program

**Support Vector Machine (SVM)** finds the maximum margin linear separator by solving the quadratic program that corresponds to $J(\theta, \theta_0)$. In the realizable case, if we disallow any margin violations, the quadratic program we have to solve is
$$
\text{Find } \theta, \theta_0 \text{ that}~~~~~~~~~~~~~~~~~~~~~~~~\\
\text{ minimize } \frac{1}{2} \|\theta\|^2 \text{ subject to}\\
~~~~~~~~~~~~~~~~~~~~~~y^{(i)}(\theta \cdot x^{(i)} + \theta _0) \geq 1, ~i = 1, ..., n\\
$$
Note that there are **infinitely** many $(\theta, \theta_0)$ that satisfy $y^{(i)} (\theta \cdot x^{(i)} + \theta _0) >= 1$ for $i = 1, ..., n$, since there is no other constraint and $\theta$ and $\theta_0$ are continuous.

#### *Explain the relationship with the minimizing the objective function:

$$
J(\theta , \theta _0) = \frac{1}{n} \sum _{i=1}^{n} \text {Loss}_ h (y^{(i)} (\theta \cdot x^{(i)} + \theta _0 )) + \frac{\lambda }{2} \mid \mid \theta \mid \mid ^2
$$

In the realizable case, we can always find a decision boundary such that the Hinge Loss term is 0. Thus the objective function is reduced to $\frac{\lambda }{2} \mid \mid \theta \mid \mid ^2$, and our goal of minimizing $J$ is reduced to find $\theta$ that minimize $\frac{1}{2} \mid \mid \theta \mid \mid ^2$.

> #### Summary
>
> * **Learning problems** can be formulated as **optimization problems** of the form: **Loss + Regularization**
> * **Linear, large margin classification**, along with many other learning problems, can be solved with **stochastic gradient descent** algorithms.
> * **Large margin linear classifier** can be also obtained via solving a **quadratic program** (SVM).

