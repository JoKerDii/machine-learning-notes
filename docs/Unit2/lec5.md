# Linear Regression

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



