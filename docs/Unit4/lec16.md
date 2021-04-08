# Mixture Models; EM algorithm

There are 4 topics and 3 exercises.

## 1. MLE under Gaussian Noise

Let $Y_ i = \theta + N_ i, i=1,\dots ,n$ where $\theta$ is an unknow parameter and $N_i$ are i.i.d. Gaussian random variables with zero mean. Upon observing $Y_i$'s, the MLE of $\theta$ is
$$
\hat{\theta } = \frac{\sum _{i=1}^{n} Y_ i}{n}
$$
As long as $N_i$ are **i.i.d., zero-mean Gaussian random variables with the same variance**, the MLE for $θ$ does not change.

> #### Exercise 37
>
> Would the ML estimator of $\theta$ change if the $N_i$'s are **independent** Gaussians with **possibly different variances** $\sigma _1^2,\dots ,\sigma _ n^2$ but **same zero mean**? Assume that $\sigma_i^2$ are **known** constants.
>
> > **Answer**: Yes, it would change.
>
> > **Solution**: 
> >
> > The **log-likelihood** (with possibly different variances) is 
> > $$
> > \log P(Y_1,\dots ,Y_ n | \theta , \sigma _1^2,\dots ,\sigma _ n^2) = -\frac{1}{2}\sum _{i=1}^{n} \log (2\pi \sigma _ i^2) - \sum _{i=1}^{n} \frac{(Y_ i - \theta )^2}{2 \sigma _ i^2}
> > $$
> > Take the derivative of the log-likelihood with respect to $\theta$ and set it to zero, then we got
> > $$
> > \sum _ i \frac{Y_ i}{\sigma _ i^2} = \theta \sum _ i \frac{1}{\sigma _ i^2}
> > $$
> > which yields 
> > $$
> >  \hat{\theta } = \frac{\sum _ i \frac{Y_ i}{\sigma _ i^2}}{\sum _ i \frac{1}{\sigma _ i^2}}
> > $$
> > which is not necessarily the same estimator as before. 

## 2. Gaussian Mixture Model (GMM)

A **Gaussian Mixture Model (GMM**) for data $x \in \R^d$:

* $K$: number of **mixture components**
* A $d$-dimensional Gaussian $\mathcal{N}(\mu^{(i)}, \sigma^2_j)$ for every $j = 1,...,K$.
* $p_1, ..., p_K$: **mixture weights**
* The parameters of a $K$- component GMM can be collectively represented as $\theta = \{p_1, ..., p_K, \mu^{(1)},...,\mu^{(K)}, \sigma_1^2, ..., \sigma_K^2\}$, which is the estimation output of this model. 
  * A mean $\mu^{(j)}$ is a $d$-dimensional vector for every $j = 1,...,K$. It is defined as **centers**.
  * A covariance $\Sigma$ that defines its **width**. This would be equivalent to the dimensions of an ellipsoid in a multivariate scenario
  * A mixing probability $p$ defines how big or small the Gaussian function will be.

Note that we assume

* **same variance** $\sigma_j^2$ across all components of the $j^{th}$ Gaussian mixture component for $j = 1, ..., K$.
* every Gaussian component is assumed to have a **diagonal covariance matrix**. (which can be generalized to a general covariance matrix)

The **likelihood** of a point $\mathbf{x}$ in a GMM is given as (according to the law of total probability)
$$
p(\mathbf{x} \mid \theta ) = \sum _{j = 1}^ K p_ j \mathcal{N}(\mathbf{x},\mu ^{(j)}, \sigma _ j^2)
$$
Note that with the above likelihood, the **posterior probability** that $\mathbf{x}$ belongs to a Gaussian component $j$ can be computed using **Bayes' Rule**.

Understanding generative model:

* first select the component $j \in \{1,...,K\}$, which is modeled using the multinomial distribution with parameters $p_1, ..., p_K$, 
* then select a point $\mathbf{x}$ from the Gaussian component $\mathcal{N}(\mu^{(j)}, \sigma_j^2)$, where $\mu^{(j)} \in \R^d$ and $\sigma_j^2$ is $d \times d$ and $\sigma_j^2 = \sigma_j^2 \begin{bmatrix} 1 & & &\\ &1 & & \\ & &...& \\& & & 1\end{bmatrix}$.

## 3. Mixture Model - Observed Case

The **global likelihood** for all points in a sample $S_n = \{\mathbf{x_1}, ..., \mathbf{x_n}\}$ is 
$$
p(S_n \mid \theta ) = \prod^n_{i=1} \sum _{j = 1}^ K p_ j \mathcal{N}(\mathbf{x},\mu ^{(j)}, \sigma _ j^2I)
$$
In obeserved case, the **indicator** function is defined as
$$
\delta(j|i) = \begin{cases} 1, &\text{if }x^{(i)} \text{ is assigned to }j\\0, & \text{otherwise }\end{cases}
$$
Take the log of the likelihood,
$$
\begin{aligned}
\log p(S_n|\theta) &= \sum^n_{i=1} [\sum^K_{j=i} \delta(j|i) \log p_j \mathcal{N}(x^{(i)}, \mu^{(j)}, \sigma^2_j I)]\\ &= \sum^K_{j=i}[\sum^n_{i=1}\delta(j|i) \log p_j \mathcal{N}(x^{(i)}, \mu^{(j)}, \sigma^2_j I)]
\end{aligned}
$$
Take the derivative and solve for parameters, then we get
$$
\begin{aligned}
\hat{n}_j &= \sum^N_{i=1}\delta(j|i)\\
\hat{p}_j &= \frac{\hat{n}_j}{n}\\
\hat{\mu}^{(j)} &= \frac{1}{\hat{n}_j} \sum^n_{i=1} \delta(j|i) \cdot x^{(i)}\\
\hat{\sigma}_j^2 &= \frac{1}{\hat{n}_j^d} \sum^n_{i=1} \delta(j|i) \cdot \|x^{(i)} - \mu^{(j)}\|^2
\end{aligned}
$$

## 4. Mixture Model - Unobserved Case: EM Algorithm

#### The Expectation Maximization (EM) Algorithm

We observe $n$ data points $\mathbf{x}_1, ..., \mathbf{x}_n$ in $\R^d$. We wish to maximize the GMM likelihood with respect to the parameter set $\theta = \{p_1, ..., p_K, \mu^{(1)}, ..., \mu^{(K)}, \sigma_1^2, ..., \sigma_K^2\}$ . The **EM algorithm** is an iterative algorithm that finds a locally optimal solution $\hat{θ}$ to the GMM likelihood maximization problem.

**Initialization:**

* One way: **random** initialization
* Another way: use **k-means** to find the initial cluster centers of $K$ clusters and use the **global variances** of the dataset as the initial variance of all the $K$ clusters. In this case, the mixture weights can be initialized to the **proportion of data points** in the clusters as found by the k-means algorithm.

**E Step:**

Find the **posterior probability** that point $\mathbf{x}^{(i)}$ was generated by cluster $j$, for every $i = 1, ..., n$ and $j = 1,...,K$. We find the **posterior** using the following equation
$$
p(\text {point }\mathbf x^{(i)}\text { was generated by cluster }j | \mathbf x^{(i)}, \theta ) \triangleq p(j \mid i) = \frac{p_ j \mathcal{N}\left(\mathbf x^{(i)}; \mu ^{(j)},\sigma _ j^2 I\right)}{p(\mathbf x^{(i)} \mid \theta )}
$$
**M Step:**

Maximize a proxy function $\hat{\ell }(\mathbf x^{(1)},\dots ,\mathbf x^{(n)} \mid \theta )$ of the log-likelihood over $\theta$, where
$$
\hat{\ell }(\mathbf x^{(1)},\dots ,\mathbf x^{(n)} \mid \theta ) \triangleq \sum _{i=1}^{n} \sum _{j = 1}^ K p(j \mid i) \log \left( \frac{p\left( \mathbf x^{(i)} \text { and } \mathbf x^{(i)} \text { generated by cluster }j \mid \theta \right)}{p(j \mid i)} \right)
$$
This is done by maximizing over $\theta$ the actual log-likelihood
$$
\ell (\mathbf x^{(1)},\dots ,\mathbf x^{(n)} \mid \theta ) = \sum _{i=1}^{n} \log \left[\sum _{j = 1}^ K p\left( \mathbf x^{(i)} \text { generated by cluster }j \mid \theta \right)\right]
$$
Take the derivative and set them to zero, so we solve for those parameters
$$
\begin{aligned}
\widehat{\mu ^{(j)}} &= \frac{\sum _{i = 1}^ n p (j \mid i) \mathbf x^{(i)}}{\sum _{i=1}^ n p(j \mid i)}\\
\widehat{p_ j} &= \frac{1}{n}\sum _{i = 1}^ n p(j \mid i)\\
\widehat{\sigma _ j^2} &= \frac{\sum _{i = 1}^ n p(j \mid i) \|  \mathbf x^{(i)} - \widehat{\mu ^{(j)}} \| ^2}{d \sum _{i = 1}^ n p(j \mid i)}
\end{aligned}
$$
The E and M steps are repeated iteratively until there is no noticeable change in the actual likelihood computed after M step using the newly estimated parameters or if the parameters do not vary by much.

> #### Exercise 38
>
> Assume that the initial means and variances of two clusters in a GMM are as follows: $\mu ^{(1)} = -3, \mu ^{(2)} = 2, \sigma _1^2=\sigma _2^2=4$. Let $p_1 = p_2 = 0.5$. Let $x^{(1)} = 0.2,x^{(2)} = -0.9,x^{(3)} = -1,x^{(4)} = 1.2,x^{(5)} = 1.8$ be five points that we wish to cluster. Please compute:
>
> * posterior probabilities $p(1|1),p(1|2),p(1|3),p(1|4),p(1|5)$ 
> * updated parameters $\hat{p}_1,\hat{\mu}_1,\hat{\sigma}^2_1$ corresponding to cluster 1.
> * log-likelihood of the data $\hat{\ell }(\mathbf x^{(1)},\dots ,\mathbf x^{(5)} \mid \theta )$.
>
> > **Answer**: 
> >
> > $p(1|1) = 0.294215,\\p(1|2)=0.6224593,\\p(1|3)=0.6513549,\\p(1|4)=0.1066906,\\p(1|5)=0.05340333\\\hat{p}_1=0.3456246,\\ \hat{\mu}_1=-0.5373289,\\ \hat{\sigma}^2_1=0.5757859\\\hat{\ell }(\mathbf x^{(1)},\dots ,\mathbf x^{(5)} \mid \theta ) = -11.64849$
>
> > **Solution**:
> >
> > The posterior formula is given as
> > $$
> > p(\text {point }\mathbf x^{(i)}\text { was generated by cluster }j | \mathbf x^{(i)}, \theta ) \triangleq p(j \mid i) = \frac{p_ j \mathcal{N}\left(\mathbf x^{(i)}; \mu ^{(j)},\sigma _ j^2 I\right)}{p(\mathbf x^{(i)} \mid \theta )}
> > $$
> > which can be written as
> > $$
> > \frac{p_ j \mathcal{N}\left(\mathbf x^{(i)}; \mu ^{(j)},\sigma _ j^2 I\right)}{p(\mathbf x^{(i)} \mid \theta )} = \frac{p_ j \mathcal{N}\left(\mathbf x^{(i)}; \mu ^{(j)},\sigma _ j^2 I\right)}{\sum_j p_ j \mathcal{N}\left(\mathbf x^{(i)}; \mu ^{(j)},\sigma _ j^2 I\right)}
> > $$
> > Therefore
> > $$
> > p(1|1) = \frac{p_1 \mathcal{N}(x^1; \mu^1, \sigma_1^2)}{p_1 \mathcal{N}(x^1; \mu^1, \sigma_1^2) + p_2 \mathcal{N}(x^1; \mu^2, \sigma_2^2)}\\
> > p(1|2) = \frac{p_1 \mathcal{N}(x^2; \mu^1, \sigma_1^2)}{p_1 \mathcal{N}(x^2; \mu^1, \sigma_1^2) + p_2 \mathcal{N}(x^2; \mu^2, \sigma_2^2)}\\
> > p(1|3) = \frac{p_1 \mathcal{N}(x^3; \mu^1, \sigma_1^2)}{p_1 \mathcal{N}(x^3; \mu^1, \sigma_1^2) + p_2 \mathcal{N}(x^3; \mu^2, \sigma_2^2)}\\
> > p(1|4) = \frac{p_1 \mathcal{N}(x^4; \mu^1, \sigma_1^2)}{p_1 \mathcal{N}(x^4; \mu^1, \sigma_1^2) + p_2 \mathcal{N}(x^4; \mu^2, \sigma_2^2)}\\
> > p(1|5) = \frac{p_1 \mathcal{N}(x^5; \mu^1, \sigma_1^2)}{p_1 \mathcal{N}(x^5; \mu^1, \sigma_1^2) + p_2 \mathcal{N}(x^5; \mu^2, \sigma_2^2)}
> > $$
> > where the Gaussian PDF $\mathcal{N}(x^i; \mu^j, \sigma_j^2)$ is
> > $$
> > \mathcal{N}(x^i; \mu^j, \sigma_j^2) = \frac{1}{\sigma_j \sqrt{2 \pi}} \exp\{-\frac{1}{2} (\frac{x^j - \mu_j}{\sigma_j})^2\}
> > $$
> > After we obtain the posterior probability corresponding to cluster 1, we can update those three parameter with the formulas
> > $$
> > \begin{aligned}
> > \widehat{\mu ^{(j)}} &= \frac{\sum _{i = 1}^ n p (j \mid i) \mathbf x^{(i)}}{\sum _{i=1}^ n p(j \mid i)}\\
> > \widehat{p_ j} &= \frac{1}{n}\sum _{i = 1}^ n p(j \mid i)\\
> > \widehat{\sigma _ j^2} &= \frac{\sum _{i = 1}^ n p(j \mid i) \|  \mathbf x^{(i)} - \widehat{\mu ^{(j)}} \| ^2}{d \sum _{i = 1}^ n p(j \mid i)}
> > \end{aligned}
> > $$
> > That is
> > $$
> > \begin{aligned}
> > \hat{\mu}_1 &= \frac{p(1|1)\times x^1+p(1|2)\times x^2+p(1|3)\times x^3+p(1|4)\times x^4+p(1|5)\times x^5}{p(1|1)+p(1|2)+p(1|3)+p(1|4)+p(1|5)}\\
> > \hat{p}_1 &= \frac{p(1|1)+p(1|2)+p(1|3)+p(1|4)+p(1|5)}{5}\\
> > \hat{\sigma_1^2} &= \frac{p(1|1)\times (x^1-\hat{\mu}_1)^2+p(1|2)\times (x^2-\hat{\mu}_1)^2+p(1|3)\times (x^3-\hat{\mu}_1)^2+p(1|4)\times (x^1-\hat{\mu}_4)^2 +p(1|5)\times (x^5-\hat{\mu}_1)^2}{p(1|1)+p(1|2)+p(1|3)+p(1|4)+p(1|5)}
> > \end{aligned}
> > $$
> > Follow the formula of log-likelihood
> > $$
> > \ell (\mathbf x^{(1)},\dots ,\mathbf x^{(n)} \mid \theta ) = \sum _{i=1}^{n} \log \left[\sum _{j = 1}^ K p\left( \mathbf x^{(i)} \text { generated by cluster }j \mid \theta \right)\right]
> > $$
> > In this case the log-likelihood is computed as
> > $$
> > \ell (\mathbf x^{(1)},\dots ,\mathbf x^{(n)} \mid \theta ) = \sum_{i = 1}^5\log[p_1 \mathcal{N}(x^i; \mu^1, \sigma_1^2) + p_2 \mathcal{N}(x^i; \mu^2, \sigma_2^2)]
> > $$



> #### Exercise 39
>
> Which of the following statements are true?
>
> A. A Gaussian mixture model can provide information about how likely it is that a given point belongs to each cluster.
>
> B. The EM algorithm converges to the same estimate of the parameters irrespective of the initialized values.
>
> C. An iteration of the EM algorithm is computationally more expensive (in terms of order complexity) when compared to an iteration of the K-means algorithm for the same number of clusters.
>
> > **Answer**: A
>
> > **solution**: 
> >
> > A. The estimated **posterior probabilities** tell us how likely it is that a given point belongs to each cluster.
> >
> > B. The EM algorithm is guaranteed (under some conditions) to only **converge** **locally**.
> >
> > C. We can see that the E-step of the algorithm takes $O(nKd)$ computations and the M-step of the algorithm is also of the order $O(nKd)$.



# Additional Readings

Relevant materials from 

http://www.cs.cmu.edu/~aarti/Class/10701/

And also other excellent course materials from

http://www.cs.cmu.edu/~aarti/