# Linear Regression

There are 6 topics and 1 exercise.

### Topics

* Objective
* Learning algorithm
  * Gradient-based
  * Closed form-based
* Regularization



## 1. Empirical Risk

**Linear Regression**: The observed value $y$ is a real number i.e. $y \in \R$. The predictor $f$ is a linear function of the feature vectors. i.e. $f(x) = \sum^d_{i=1} \theta_i x_i + \theta_0$.

**Empirical Risk:**
$$
R_ n(\theta ) = \frac{1}{n} \sum _{t=1}^{n} \text {Loss}(y^{(t)} - \theta \cdot x^{(t)})
$$
where the loss function could be 

* Hinge Loss
  $$
  \text {Loss}_ h(z) = \begin{cases}  0 & \text {if } z \geq 1 \\ 1 -z, & \text { otherwise} \end{cases}.
  $$

* Squared Error Loss
  $$
  \displaystyle \text {Loss}(z) = \frac{z^2}{2}.
  $$

## 2. Error Decomposition and The Bias-Variance Trade-Off

### A more formal definition of linear regression problem:

From observed training set $(x_1, y_1), (x_2, y_2), ..., (x_i, y_i)$, we hope to find a function $\hat{f}$ to approximate the true function $f$ which describes the relationship between random variables $x \in \R^d$ and $y \in \R$.
$$
y = f(x) + \epsilon
$$
where $\epsilon \sim \mathcal{N}(0,1)$.

Here we assume a random noise variable $\epsilon$ to be added to $y$ because our observed data might not be 100% accurate as there can be many kinds of noise and uncertainty containing in the data. 

### Derive the bias-variance trade-off

We can find $\hat{f}$ by minimizing the empirical risk $R_n(\theta)$ on the training set, however, the training set is a random observation of the underlying relationship and contains noise, which means that different training set will give different estimator $\hat{f}(x)$. Therefore, we define $E[\hat{f}(x)]$ to be **expected estimator over all possible training sets**.
$$
\begin{aligned}
\mathbb {E}[(y-\hat{f}(x))^2] & =  \mathbb {E}[(f(x) + \epsilon -\hat{f}(x))^2] \\ & =  (f(x)-\mathbb {E}[\hat{f}(x)])^2 + \mathbb {E}[(\hat{f}(x)-\mathbb {E}[\hat{f}(x)])^2] + \mathbb {E}[\epsilon ^2]
\end{aligned}
$$

1. **Bias term**: $(f(x)-\mathbb {E}[\hat{f}(x)])^2$

   It describes how much the average estimator fitted over all datasets deviates from the underlying true $f(x)$. It is also known as **structural mistake**.

2. **Variance term**: $\mathbb {E}[(\hat{f}(x)-\mathbb {E}[\hat{f}(x)])^2]$

   It describes on average how much a single estimator deviates from the expected estimator over all data sets. It is also called **estimator error**.

3. **Error term**: $\mathbb {E}[\epsilon ^2]=\sigma ^2$

   It is the error from the inherent noise  of the data and we can do nothing to minimize it, thus it is sometimes called **irreducible error**.

The task of supervised learning is to reduce the bias and variance simultaneously , but it is not possible because of the noise in the training data.

### Property

* underfitting - simple model - more bias - less variance
* overfitting - complex model - less bias - more variance

![bias_variance](../assets/images/U2/bias_variance.png)



## 3. Gradient Based Approach

Compute the gradient of least squared loss function as an example:
$$
\nabla _{\theta } (y^{(t)} - \theta x^{(t)})^2/2 = (y^{(t)} - \theta x^{(t)}) \nabla _{\theta } (y^{(t)} - \theta x^{(t)}) = (y^{(t)} - \theta x^{(t)}) x^{(t)}
$$
The update rule is:
$$
\theta = \theta + \eta (y^{(t)} - \theta x^{(t)}) x^{(t)}
$$

> #### Exercise 17
>
> Let $Rn(θ)$ be the least squares criterion defined by
> $$
> \displaystyle  R_ n(\theta )=\frac{1}{n} \sum _{t=1}^{n} \text {Loss}\left(y^{(t)} - \theta \cdot x^{(t)}\right)
> $$
> Which of the following is true? Choose all those apply.
>
> A. The least squares criterion $Rn(θ)$ is a sum of functions, one per data point.
>
> B. Stochastic gradient descent is slower than gradient descent.
>
> C. $∇θRn(θ)$ is a sum of functions, one per data point.
>
> > **Answer**: AC
>
> > **Solution**: 
> >
> > AC: For every point, the loss is a function of $\theta$, so the least squares criterion $R_n(\theta)$ is a sum of functions, one per data point, and thus $\nabla _{\theta } R_ n(\theta )$ is also a sum of functions one per data point.
> >
> > B: SGD is faster than gradient descent, that's why SGD is favorable.



## 4. Closed Form Solution

Here we have closed form solution because the empirical risk function happens to be a convex function. However, usually in most of the machine learning problems this would not happen.

Computing the gradient of
$$
R_ n(\theta ) = \frac{1}{n} \sum _{t=1}^{n} \frac{(y^{(t)} - \theta \cdot x^{(t)})^2}{2},
$$

Since 
$$
\nabla R_ n(\theta )  = \frac{1}{n} \sum^n_{t=1} y^{(t)} x^{(t)} + \frac{1}{n} \sum^n_{t=1} \hat{\theta} \cdot x^{(t)} x^{(t)}
$$
As the dot product $\hat{\theta} \cdot x^{(t)}$ is a scalar, we can simply move it around
$$
\nabla R_ n(\theta )  = \frac{1}{n} \sum^n_{t=1} y^{(t)} x^{(t)} + \frac{1}{n} \sum^n_{t=1} x^{(t)}\hat{\theta} \cdot x^{(t)}
$$
Note that $\theta \cdot x = x^T \theta$, we get
$$
\nabla R_ n(\theta )  = \frac{1}{n} \sum^n_{t=1} y^{(t)} x^{(t)} + \frac{1}{n} \sum^n_{t=1} x^{(t)} (x^{(t)})^T \hat{\theta}
$$

Finally we get: 
$$
\nabla R_ n(\theta ) = A\theta - b (=0) \quad \text {where } \,  A = \frac{1}{n} \sum _{t=1}^{n} x^{(t)} ( x^{(t)})^ T,\,  b = \frac{1}{n} \sum _{t=1}^{n} y^{(t)} x^{(t)}.
$$

To compute $\hat{\theta}$, we solve
$$
\begin{aligned}
A\theta & = b \\
\hat{\theta} & = A^{-1} b\\
\end{aligned}
$$
The **problem** here are

* Matrix $A$ is not always invertible so that it does not always have a unique solution. [Note that $A$ is invertible if vectors $x^{(1)}, ...,x^{(n)}$ span $R^d$, (n>>d)]. 
* We must have enough training set for this operation to work. 
* The computational cost is $O(d^3)$ which is very high.



## 5. Generalization and Regularization

The loss function is defined as
$$
J_{n, \lambda } (\theta , \theta _0) = \frac{1}{n} \sum _{t=1}^{n} \frac{(y^{(t)} - \theta \cdot x^{(t)})^2}{2} + \frac{\lambda }{2} \left\|  \theta  \right\| ^2
$$
Apply gradient descent:
$$
\nabla_\theta (\frac{\lambda}{2} \| \theta\|^2 + (y^{(t)} - \theta x^{(t)})^2/2) = \lambda \theta - (y^{(t)} - \theta x^{(t)}) x^{(t)}
$$
The update rule is 
$$
\theta = \theta - \eta(\lambda \theta - (y^{(t)} - \theta x^{(t)})x^{(t)}) = (1-\eta \lambda) \theta + \eta(y^{(t)} - \theta x^{(t)}) x^{(t)}
$$

## 6. Equivalence of regularization to a Gaussian Prior on Weights

The regularized linear regression can be interpreted from a probabilistic point of view. Suppose we are fitting a linear regression model with $n$ data points $(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$. Let's assume the ground truth is that $y$ is linearly related to $x$ but we also observed some noise $ϵ$ for $y$:
$$
y_ t=\theta \cdot x_ t + \epsilon
$$
where $\epsilon \sim \mathcal{N}(0,\sigma ^2)$.

Then the likelihood of our observed data is
$$
\prod _{t=1}^{n} \mathcal{N}(y_ t|\theta x_ t, \sigma ^2).
$$
Now, if we impose a Gaussian prior $\mathcal{N}(\theta |0,\lambda ^{-1})$, the likelihood will change to
$$
\prod _{t=1}^{n} \mathcal{N}(y_ t|\theta x_ t, \sigma ^2)\mathcal{N}(\theta |0, \lambda ^{-1}).
$$
Take the logarithm of the likelihood, we will end up with
$$
\sum _{t=1}^{n} -\frac{1}{2 \sigma ^2}(y_ t-\theta x_ t)^2-\frac{1}{2}\lambda \| \theta \| ^2 + \text {constant}.
$$
Take the derivative and set it to 0:
$$
\frac{1}{\sigma^2} (y_t - \theta x_t) x_t - \lambda \theta = 0
$$
We get
$$
\hat{\theta} = \frac{y_t x_t}{x_t x_t^T + \sigma^2 \lambda}
$$
Thus we can conclude that maximizing this loglikelihood equivalent to minimizing the regularized loss in the linear regression.

If $\lambda \rightarrow \infty$, $\hat{\theta} \rightarrow 0$, which is an underfitting horizontal line.

> Question: 
>
> What does larger $λ$ mean in this probabilistic interpretation?  (Think of the error decomposition)



# Additional Readings

A good lecture note of Tuo Zhao - Assistant Professor of CSE at Gatech

https://www2.isye.gatech.edu/~tzhao80/Lectures/8803.pdf