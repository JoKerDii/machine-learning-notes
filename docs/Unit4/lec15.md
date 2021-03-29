# Generative Models

There are 7 topics and 4 exercises

## 1. Generative vs. Discriminative models

**Generative models** work by explicitly modelling the probability distribution of each of the individual classes in the training data. 

* e.g. **Gaussian generative models** fit a Gaussian probability distribution to the training data in order to estimate the probability of a new data point belonging to different classes during prediction.

**Discriminative models** learn explicit decision boundary between classes. 

* e.g. **SVM classifier** which is a discriminative model learns its decision boundary by maximizing the distance between training data points and a learned decision boundary.

## 2. Simple Multinomial Generative model

Consider a very simple multinomial model $M$ to generate one word $w$ at a time independently in a document with a fixed vocabulary $W$. So $M$ models the probability a word $w \in W$ is generated, which can be denote as $P(w | \theta ) = \theta _ w$, where $\theta_w$ is a parameter in our model $M$ indicating the probability of a model choosing the word $w$.

Note that since $\theta_w$ is a probability, $0 \le \theta _ w \le 1$ and $\sum _{w\in W} \theta _ w = 1$

## 3. Maximum Likelihood Estimate (MLE)

The **likelihood function** is the product of the probabilities of words from document $D$.
$$
P(D|\theta) = \prod^n_{i=1} \Theta_{w_i} = \prod_{w \in W} \theta_w^{\text{count}(w)}
$$
The **maximum likelihood estimate (MLE)** of $\theta$ is the value of $\theta$ that maximizes the likelihood function or the log of the likelihood function.
$$
\max_{\theta} P(D|\theta) = \max_{\theta} \prod_{w \in W} \theta_{w}^{\text{count}(w)}
$$
We take the natural logarithm of both sides for computational convenience
$$
\text{log} P(D|\theta) = \text{log} \prod_{w \in W} \theta_w^{\text{count}(w)} = \sum_{w \in W} \text{count}(w) \text{log} \theta_w
$$
To maximize $\text{log}P(D|\theta)$ subject to the constraint $\sum_{w \in W} \theta_W = 1$, we use **Lagrange multiplier method**.

Define the Lagrange function as:
$$
L = \log P(D | \theta ) + \lambda \left(\sum _{w \in W} \theta _ w - 1\right)
$$
where $\lambda$ is a constant scalar. Then find the stationary points of $L$ by solving the equation $\nabla _\theta L=0$. The components of this equation are
$$
\frac{\partial }{\partial \theta _ w} \left(\log P(D | \theta ) + \lambda \left(\sum _{w \in W} \theta _ w - 1\right)\right) = 0 \qquad \text {for all } w \in W.
$$
Solve the above equation by
$$
\frac{\partial \log P(D | \theta _ w)}{\partial \theta _ w} + \lambda = 0\\
\frac{\partial \log \prod _{w \in W} \theta _ w^{\text {count}(w)}}{\partial \theta _ w} + \lambda = 0\\
\frac{\partial \sum _{w \in W} \log \theta _ w\times \text {count}(w)}{\partial \theta _ w} + \lambda = 0\\
\frac{\text {count}(w)}{\theta _ w} + \lambda = 0
$$
The solution for $\theta_w$ is 
$$
\theta_w = \frac{-\text{count(w)}}{\lambda}
$$
We apply $\sum_{w \in W} \theta_W = 1$, and solve for $\lambda$
$$
\sum_{w \in W} \theta_w = 1\\
\sum_{w \in W}  \frac{-\text{count(w)}}{\lambda} = 1\\
\sum_{w \in W} \text{count(w)} = -\lambda\\
\lambda = -\sum_{w \in W}\text{count} (w)\\
$$
Substitute $\lambda$ back to $\theta_w$ we have
$$
\theta _ w = \frac{\text {count}(w)}{\sum _{w \in W} \text {count}(w)}
$$

#### Recall Method of Lagrange Multipliers

Without the constraint, the optimization problem can be solved by setting the gradient of $f$ to zero
$$
\nabla f = 0
$$
With the constraint, we can solve the equation by
$$
\nabla f = \lambda \nabla g
$$
where $\lambda$ is a constant scalar. 

Since the equation $\nabla f = \lambda \nabla g$ is equivalent to $\nabla L = 0$ where $L = f - \lambda g$, the problem of optimizing $f$ subject to $g = 0$ can be reformulated as optimizing the function $L$ along with the constraint $g = 0$. The function $L$ is called **Lagrangian function**, and the scalar $\lambda$ is the **Lagrange multiplier**.

> #### Exercise 33
>
> Suppose a document contains two words $W_{\theta} = \{0,1\}$ where $\theta_0 = \theta, \theta_1 = 1 - \theta$. What is the MLE of model parameter？
>
> > **Answer**:  $\hat{\theta} = \frac{\text{count}(0)}{\text{count}(0) + \text{count}(1)}$
>
> > **Solution**: 
> >
> > The log of likelihood function is
> > $$
> > \sum_{w \in W} \text{count}(w) \text{log} \theta_w = \text{count}(0) \text{log} \theta + \text{count}(1) \text{log}(1-\theta)
> > $$
> > Take the derivative and set it to zero,
> > $$
> > \frac{\partial {(\text{count}(0) \text{log} \theta + \text{count}(1) \text{log}(1-\theta))}}{\partial \theta} = \frac{\text{count}(0)}{\theta} + \frac{\text{count}(1)}{1-\theta}= 0
> > $$
> >
> > We got
> > $$
> > \hat{\theta} = \frac{\text{count}(0)}{\text{count}(0) + \text{count}(1)}
> > $$
> >

> #### Exercise 34
>
> Suppose we us a multinomial model $M$ to generate documents that are English letter sequences $W = \{a,b,c,...,z\}$.  What is the minimal number of parameters that the model $M$ should have? 
>
> > **Answer**: 25
>
> > **Solution**: For a multinomial generative model we should have a parameter $\theta_w$ for each word $w \in W$, but since the parameters should sum up to one, we can express one of the parameters as 1 minus the sum of all others. 

> #### Exercise 35
>
> If we want to use MLE to fit the parameter $\lambda$ with the training data $x_1, x_2, ..., x_n$, what is the MLE for Poisson Distribution?
>
> The PMF of Poisson distribution is 
> $$
> P(X=x)=\frac{\lambda ^ x e^{-\lambda }}{x!}
> $$
>
> > **Answer**: $\frac{1}{n}\sum _ i x_ i$.
>
> > **Solution**: 
> >
> > Compute the log likelihood
> > $$
> > \begin{aligned}
> > \log \prod _ i P(X=x_ i) &= \log \prod _ i \frac{\lambda ^{x_ i} e^{-\lambda }}{x_ i!}\\
> > &= \sum _ i \log (\lambda ^{x_ i}) + \log (e^{-\lambda })-\log (x_ i!)\\
> > &= \log \lambda \sum _ i x_ i -n\lambda - \sum _ i \log (x_ i!)
> > \end{aligned}
> > $$
> > Take the derivative, set it to zero, and compute MLE for $\lambda$ 
> > $$
> > \frac{1}{\lambda }\sum _ i x_ i -n = 0\\
> > \lambda = \frac{1}{n}\sum _ i x_ i
> > $$
> > This is in accordance with the fact that $λ$ is the expectation of a Poisson variable with parameter $\lambda$.

## 4. Predictions of a Generative Multinomial Model

Consider using a multinomial generative model $M$ for the task of binary classification of positive (+) and negative (-) classes. Let the parameters of $M$ that maximize the likelihood of training data for the positive class be denoted by $\theta^+$ and for the negative class be denoted by $\theta^-$. 

Suppose we classify a new document $D$ to belong to positive class if
$$
\log \frac{P(D | \theta ^+)}{P(D | \theta ^-)} \ge 0
$$
where $P(D | \theta )$ stands for the probability that document $D$ is generated using a multinomial distribution with parameters $\theta$. 

Equivalently, the decision boundary can be written as
$$
\begin{aligned}
\log \frac{P(D | \theta ^+)}{P(D | \theta ^-)} & = \log P(D | \theta ^+) - \log P(D | \theta ^-)\\
& = \log \prod_{w \in W}\theta^+_{count(w)} - \log \prod_{w \in W}\theta^-_{count(w)}\\
& = \sum_{w \in W}\text{count}(w)\log\theta^+ - \sum_{w \in W}\text{count}(w)\log\theta^-\\
& = \sum_{w \in W}\text{count}(w)\log \frac{\theta^+_w}{\theta^-_w} \\
& = \sum_{w \in W}\text{count}(w) \theta'_w \\

\end{aligned}
$$
where $\theta '_ w = \log \frac{\theta _ w^+}{\theta _ w^-}$. This shows that the generative classifier $M$ is equivalent to a **linear classifier**.

> #### Exercise 36: 
>
> Let $W = \{\text{Thor, Loki, Hulk}\}$. 0 represent positive, 1 represents negative.
>
> Let $p(\text{Thor}|0) = p(\text{Loki}|0) = p(\text{Hulk}|0) = 1/3$ and let $p(\text{Thor}|1) = p(\text{Loki}|1) = 1/4$ and $p(\text{Hulk}|1) = 1/2$. Given a document $D = \text{Thor Thor Hulk Loki Loki}$, which class (positive or negative) would you classify the document to?
>
> > **Answer**: positive
>
> > **Solution**: 
> >
> > The counts of the words are $\text{count(Thor)} = 2, \text{count(Loki)} = 2, \text{count(Hulk)} = 1$. 
> > $$
> > \begin{aligned}
> > \hat{\theta }_{\text {Thor}} &=\log \frac{\theta^0_\text {Thor}}{\theta^1_\text {Thor}}= log(\frac{4}{3}) \approx 0.124939\\
> > \hat{\theta }_{\text {Loki}} &=\log \frac{\theta^0_\text {Loki}}{\theta^1_\text {Loki}}= log(\frac{4}{3}) \approx 0.124939\\
> > \hat{\theta }_{\text {Hulk}} &=\log \frac{\theta^0_\text {Hulk}}{\theta^1_\text {Hulk}}= log(\frac{2}{3}) \approx -0.176091\\
> > \end{aligned}
> > $$
> > Therefore, 
> > $$
> > \sum _{w \in \mathcal{W}} \text {count}(w) \hat{\theta }_ w = 4 \cdot 0.124939 - 0.176091 > 0
> > $$
> > which would classify the document to class $0(+)$.

## 5. Prior, Posterior and Likelihood

Apply **Bayes's Rule** to derive the decision boundary
$$
\begin{aligned}
P(y = +|D) &= \frac{P(D|\theta^+)P(+)}{P(D)}\\
&= \frac{P(D|\theta^+)P(+)}{P(D|\theta^+)P(+) + P(D|\theta^-)(1-P(+))}\\
\end{aligned}
$$
where $P(y = +|D)$ is the **posterior**, $P(+)$ is the **prior**.
$$
\begin{aligned}
\log \frac{P(y = + | D)}{P(y = - | D)} &= \log \frac{P(D|\theta^+)P(y = +)}{P(D|\theta^-)P(y=-)}\\ &= \log \frac{P(D|\theta^+)}{P(D|\theta^-)} + \log \frac{P(y=+)}{P(y=-)}\\
&= \sum_{w \in W} \text{count}(w) \theta'_w + \theta'_0\\
\end{aligned}
$$
where $\theta '_ w = \log \frac{\theta _ w^+}{\theta _ w^-}$ and $\theta'_0 =\log \frac{P(y=+)}{P(y=-)}$. This shows that the generative classifier $M$ is equivalent to a **linear classifier** with an intercept.

## 6. Gaussian Generative models

A random vector $\mathbf{X}=(X^{(1)},\ldots ,X^{(d)})^ T$ is a **Gaussian vector  / multivariate Gaussian / normal variable**, if any linear combination of its components is a (univariate) Gaussian variable or a constant (a Gaussian variable with zero variance), i.e., if $\alpha^TX$ is (univariate) Gaussian or constant for any constant non-zero vector $\alpha \in \R^d$.

The distribution of $\mathbf{X}$, the **d-dimensional Gaussian or normal distribution**, is completely specified by the vector mean $\mu =\mathbf E[\mathbf{X}]= (\mathbf E[X^{(1)}],\ldots ,\mathbf E[X^{(d)}])^ T$ and the $d \times d$ covariance matrix $\Sigma$. If $\Sigma$ is invertible, then the pdf of $\mathbf{X}$ is 
$$
f_{\mathbf{X}}(\mathbf x) = \frac{1}{\sqrt{\left(2\pi \right)^ d \text {det}(\Sigma )}}e^{-\frac{1}{2}(\mathbf x-\mu )^ T \Sigma ^{-1} (\mathbf x-\mu )}, ~ ~ ~ \mathbf x\in \mathbb {R}^ d
$$
where $\text{det}(\Sigma)$ is the determinant of the $\Sigma$, which is positive when $\Sigma$ is invertible.

If $\mu = 0$ and $\Sigma$ is the identity matrix, then $\mathbf{X}$ is called a **standard normal random vector**.

Therefore, the **likelihood** of $x$ being generated from a **multi-dimensional Gaussian** with mean $\mu$ and all the components being uncorrelated and having the same standard deviation $\sigma$ is 
$$
P(x | \mu , \sigma ^2) = \frac{1}{(2\pi \sigma ^2)^{d/2}} \text {exp}(-\frac{1}{2\sigma ^2} \|  x - \mu \| ^2)
$$

## 7. MLE for the Gaussian Distribution

Let $S_ n = \{ X^{(1)}, X^{(2)}, ... X^{(n)}\}$ be i.i.d random variables following a Gaussian distribution with mean $\mu$ and variance $\sigma^2$. The their **joint probability density function** is given by
$$
\prod _{t=1}^ n P(x^{(t)} | \mu , \sigma ^2) = \prod _{t=1}^ n \frac{1}{{(2 \pi \sigma ^2)^{d/2} }} e^{-{\| x^{(t)} - \mu \| }^2 / 2\sigma ^2}
$$
Taking logarithm of the above function, we get
$$
\begin{aligned}
\log P(S_ n | \mu , \sigma ^2)\,& =\, \log \left(\prod _{t=1}^ n \frac{1}{{(2 \pi \sigma ^2)^{d/2} }} e^{-{\| x^{(t)} - \mu \| }^2 / 2\sigma ^2}\right)\\ & = \sum _{t=1}^ n \log \frac{1}{{(2 \pi \sigma ^2)^{d/2} }} + \sum _{t=1}^ n \log e^{-{\| x^{(t)} - \mu \| }^2 / 2\sigma ^2}\\ & = \sum _{t=1}^ n - \frac{d}{2} \log (2 \pi \sigma ^2) + \sum _{t=1}^ n \log e^{-{\| x^{(t)} - \mu \| }^2 / 2\sigma ^2}\\
& = -\frac{nd}{2} \log (2\pi \sigma ^2) - \frac{1}{2 \sigma ^2} \sum _{t=1}^{n} \| x^{(t)} - \mu \| ^2
\end{aligned}
$$
Take the partial derivative of the function above with respect to $\mu$ and $\sigma^2$, set it to zero, and solve for $\hat{\mu}$.
$$
\begin{aligned}
\frac{\partial \log P(S_ n | \mu , \sigma ^2)}{\partial \mu } &= \frac{1}{\sigma ^2} \sum _{t=1}^{n} (x^{(t)} - \mu ) = 0\\
\frac{\partial \log P(S_ n | \mu , \sigma ^2)}{\partial \sigma ^2} &= - \frac{nd}{2\sigma ^2} + \frac{\sum _{t=1}^{n} \| x^{(t)} - \mu \| ^2}{2(\sigma ^2)^2} = 0
\end{aligned}
$$
Finally, we get
$$
\begin{aligned}
\hat{\mu } &= \frac{\sum _{t=1}^{n} x^{(t)}}{n}\\
\hat{\sigma }^2 &= \frac{\sum _{t=1}^ n \| x^{(t)} - \mu \| ^2}{nd}
\end{aligned}
$$
