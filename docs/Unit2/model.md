# Implementation Ideas

There are 7 topics and 1 exercise.

## 1. Closed Form Solution of Linear Regression

To solve the linear regression problem, the closed form solution is
$$
\theta = (X^ T X + \lambda I)^{-1} X^ T Y
$$
where $I$ is the identity matrix.

## 2. Softmax Function

For each data point $x^{(i)}$, the probability that $x^{(i)}$ is labeled as $j$ for $j = 0,1,\dots ,k-1$. The softmax function $h$ for a particular vector $x$ requires computing
$$
h(x) = \frac{1}{\sum _{j=0}^{k-1} e^{\theta _ j \cdot x / \tau }} \begin{bmatrix}  e^{\theta _0 \cdot x / \tau } \\ e^{\theta _1 \cdot x / \tau } \\ \vdots \\ e^{\theta _{k-1} \cdot x / \tau } \end{bmatrix},
$$
where $\tau > 0$ is the **temperature parameter**. When computing the output probabilities ( with range of $[0,1]$), the terms $e^{\theta _ j \cdot x / \tau }$ may be very large or very small, due to the use of the exponential function. This can cause **numerical or overflow errors**. 

To deal with this, we substract some fixed amount $x$ from each exponent to keep the resulting number from getting too large. Since substracting some fixed amount $c$ from each exponent will not change the final probabilities. 
$$
h(x) = \frac{e^{-c}}{e^{-c}\sum _{j=0}^{k-1} e^{\theta _ j \cdot x / \tau }} \begin{bmatrix}  e^{\theta _0 \cdot x / \tau } \\ e^{\theta _1 \cdot x / \tau } \\ \vdots \\ e^{\theta _{k-1} \cdot x / \tau } \end{bmatrix} \\ = \frac{1}{\sum _{j=0}^{k-1} e^{[\theta _ j \cdot x / \tau ] - c}} \begin{bmatrix}  e^{[\theta _0 \cdot x / \tau ] - c} \\ e^{[\theta _1 \cdot x / \tau ] - c} \\ \vdots \\ e^{[\theta _{k-1} \cdot x / \tau ] - c} \end{bmatrix},
$$
A suitable choice for this fixed amount is $c = \max _ j \theta _ j \cdot x / \tau$.

> > #### Exercise 23: 
> >
> > Explain how the temperature parameter affects the probability of a sample $x^{(i)}$ being assigned a label that has a large $θ$. What about a small $θ$?
> >
> > A. Larger temperature leads to less variance
> >
> > B. Smaller temperature leads to less variance
> >
> > C. Smaller temperature makes the distribution more uniform
>
> > **Answer**: B
>
> > **Solution**: 
> >
> > **The higher the $\tau$, the 'softer' the distribution will be.**  'Softer' is that the model will basically be less confident about it's prediction. For example: 
> >
> > a) Sample 'hard' softmax probs : `[0.01,0.01,0.98]`
> >
> > b) Sample 'soft' softmax probs : `[0.2,0.2,0.6]`
> >
> > In the case of 'a', which is a 'harder' distribution. The model is very confident about it's predictions. However, in many real cases, this confident model is not preferred. For example, if you are using an RNN to generate text, you are basically sampling from your output distribution and choosing the sampled word as your output token (and next input). IF your model is extremely confident, it may produce very repetitive and uninteresting text. You want it to produce more diverse text which it will not produce because when the sampling procedure is going on, most of the probability mass will be concentrated in a few tokens and thus your model will keep selecting a number of words over and over again. In order to give other words a chance of being sampled as well, you could plug in the temperature variable and produce more diverse text.
> >
> > The reason why higher temperatures lead to softer distributions is about the property of exponential function. The temperature parameter penalizes bigger logits more than the smaller logits. The exponential function is an 'increasing function'. So if a term is already big, penalizing it by a small amount would make it much smaller (% wise) than if that term was small. For example:
> >
> > We have: 
> >
> > `exp(6) ~ 403`
> >
> > `exp(3) ~ 20`
> >
> > Now let's 'penalize' this term with a temperature of 1.5:
> >
> > `exp(6/1.5) ~ 54`
> >
> > `exp(3/1.5) ~ 7.4`
> >
> > In % terms, the bigger the term is, the more it shrinks when the temperature is used to penalize it. When the bigger logits shrink more than your smaller logits, more probability mass (to be computed by the softmax) will be assigned to the smaller logits.

## 3. Cost Function

The following cost function $J(\theta)$ computes the total cost over every data point (using natural log)
$$
J(\theta ) = -\frac{1}{n}\Bigg[\sum _{i=1}^ n \sum _{j=0}^{k-1} [[y^{(i)} == j]] \log {\frac{e^{\theta _ j \cdot x^{(i)} / \tau }}{\sum _{l=0}^{k-1} e^{\theta _ l \cdot x^{(i)} / \tau }}}\Bigg] + \frac{\lambda }{2}\sum _{j=0}^{k-1}\sum _{i=0}^{d-1} \theta _{ji}^2
$$

## 4. Gradient Descent

Within $J(\theta)$ we have the probability of each label for each data point:
$$
\frac{e^{\theta _ j \cdot x^{(i)} / \tau }}{\sum _{l=0}^{k-1} e^{\theta _ l \cdot x^{(i)} / \tau }} = p(y^{(i)} = j | x^{(i)}, \theta )
$$
In order to run the gradient descent algorithm to minimize the cost function, we need to take the derivative of $J(\theta)$ with respect to a particular $\theta_m$. 

So we first compute $\frac{\partial p(y^{(i)} = j | x^{(i)}, \theta )}{\partial \theta _ m}$

When $m = j$,
$$
\frac{\partial p(y^{(i)} = j | x^{(i)}, \theta )}{\partial \theta _ m}=\frac{x^{(i)}}{\tau }p(y^{(i)} = m | x^{(i)}, \theta )[1-p(y^{(i)} = m | x^{(i)}, \theta )]
$$
When $m \neq j$,
$$
\frac{\partial p(y^{(i)} = j | x^{(i)}, \theta )}{\partial \theta _ m}=- \frac{x^{(i)}}{\tau }p(y^{(i)} = m | x^{(i)}, \theta )p(y^{(i)} = j | x^{(i)}, \theta )
$$
Now we compute
$$
\begin{aligned}
\frac{\partial }{\partial \theta _ m} \Bigg[ \sum _{j=0}^{k-1} [[y^{(i)} == j]] \log {\frac{e^{\theta _ j \cdot x^{(i)} / \tau }}{\sum _{l=0}^{k-1} e^{\theta _ l \cdot x^{(i)} / \tau }}} \Bigg]  & = \sum _{j=0, j\neq m}^{k-1} \Bigg[ [[y^{(i)} == j]] [- \frac{x^{(i)}}{\tau }p(y^{(i)} = m | x^{(i)}, \theta )] \Bigg] \\ & + [[y^{(i)} == m]] \frac{x^{(i)}}{\tau }[1-p(y^{(i)} = m | x^{(i)}, \theta )]\\ & = \frac{x^{(i)}}{\tau } \Bigg[ [[y^{(i)} == m]] - p(y^{(i)} = m | x^{(i)}, \theta ) \sum _{j=0}^{k-1} [[y^{(i)} == j]] \Bigg]\\ & = \frac{x^{(i)}}{\tau } \Bigg[ [[y^{(i)} == m]] - p(y^{(i)} = m | x^{(i)}, \theta ) \Bigg]
\end{aligned}
$$
Plug this into the derivative of $J(\theta)$ we have
$$
\begin{aligned}
\frac{\partial J(\theta )}{\partial \theta _ m} & = \frac{\partial }{\partial \theta _ m}\Bigg[-\frac{1}{n}\Bigg[\sum _{i=1}^ n \sum _{j=0}^{k-1} [[y^{(i)} == j]] \log p(y^{(i)} = j | x^{(i)}, \theta ) \Bigg] + \frac{\lambda }{2}\sum _{j=0}^{k-1}\sum _{i=0}^{d-1} \theta _{ji}^2\Bigg] \\
& = -\frac{1}{\tau n} \sum _{i = 1} ^{n} [x^{(i)}([[y^{(i)} == m]] - p(y^{(i)} = m | x^{(i)}, \theta ))] + \lambda \theta _ m
\end{aligned}
$$
To run the gradient descent, we will update $\theta$ at each step with $\theta \leftarrow \theta - \alpha \nabla _{\theta } J(\theta )$, where $\alpha$ is the learning rate.

## 5. Principal Components Analysis (PCA)

**PCA** is the most popular method for linear dimension reduction of data. It finds (orthogonal) direction of maximal variation in the data.  By projecting an $n×d$ dataset $X$  onto $k≤d$ of these directions, we get a new dataset of lower dimension that reflects more variation in the original data than any other $k$-dimensional linear projection of $X$. By going through some linear algebra, it can be proven that these directions are equal to the $k$ eigenvectors corresponding to the $k$ largest eigenvalues of the covariance matrix $\widetilde{X}^ T \widetilde{X}$, where $\widetilde{X}$ is a centered version of our original data.

#### Project onto Principal Components

Note that to project a given $n×d$ dataset $X$ into its $k$-dimensional PCA representation, one can use matrix multiplication, after first centering $X$:
$$
\widetilde{X} V
$$
where $\widetilde{X}$ is the centered original data $X$ using the mean learned from training data, and $V$ is the $d \times k$ matrix whose columns are the top $k$ eigenvectors of $\widetilde{X}^ T \widetilde{X}$. Since the eigenvectors are of unit-norm, there is no need to divide them by their length.

#### Note:

**we only use the training set to determine the principal components.** It is **improper** to use the test set for anything except evaluating the accuracy of our predictive model. If the test data is used for other purposes such as selecting good features, it is possible to overfit the test set and obtain overconfident estimates of a model's performance.

## 6. Cubic Feature

A **cubic feature** maps an input vector $x = [x_1,\dots , x_ d]$ into a new feature vector $\phi (x)$, defined so that for any $x, x' \in \mathbb {R}^ d$:
$$
\phi (x)^ T \phi (x') = (x^ T x' + 1)^3
$$

## 7. Kernel Method

In the **kernel perceptron algorithm**, the weights $\theta$ can be represented by a linear combination of features 
$$
\theta = \sum _{i=1}^{n} \alpha ^{(i)} y^{(i)} \phi (x^{(i)})
$$
In the **softmax regression formulation**, we can also apply this representation of the weights
$$
\theta _ j = \sum _{i=1}^{n} \alpha _{j}^{(i)} y^{(i)} \phi (x^{(i)})\\
h(x) = \frac{1}{\sum _{j=1}^ k e^{[\theta _ j \cdot \phi (x) / \tau ] - c}} \begin{bmatrix}  e^{[\theta _1 \cdot \phi (x) / \tau ] - c} \\ e^{[\theta _2 \cdot \phi (x) / \tau ] - c} \\ \vdots \\ e^{[\theta _ k \cdot \phi (x) / \tau ] - c} \end{bmatrix}\\
h(x) = \frac{1}{\sum _{j=1}^ k e^{[\sum _{i=1}^{n} \alpha _{j}^{(i)} y^{(i)} \phi (x^{(i)}) \cdot \phi (x) / \tau ] - c}} \begin{bmatrix}  e^{[\sum _{i=1}^{n} \alpha _{1}^{(i)} y^{(i)} \phi (x^{(i)}) \cdot \phi (x) / \tau ] - c} \\ e^{[\sum _{i=1}^{n} \alpha _{2}^{(i)} y^{(i)} \phi (x^{(i)}) \cdot \phi (x) / \tau ] - c} \\ \vdots \\ e^{[\sum _{i=1}^{n} \alpha _{k}^{(i)} y^{(i)} \phi (x^{(i)}) \cdot \phi (x) / \tau ] - c} \end{bmatrix}
$$
We need the inner product between two features after mapping $\phi (x_ i) \cdot \phi (x)$, but not the real mapping $\phi(x)$, where $x_i$ is a point in the training set and $x$ is the new data point for which we want to compute the probability. If we can create a kernel function $K(x,y) = \phi (x) \cdot \phi (y)$ for any two points $x$ and $y$, we can then kernelize our softmax regression algorithm.

#### An example of polynomial kernel

Suppose we map the features into $d$ dimensional polynomial space,
$$
\phi (x) = \langle x_ d^2, \ldots , x_1^2, \sqrt{2} x_ d x_{d-1}, \ldots , \sqrt{2} x_ d x_1, \sqrt{2} x_{d-1} x_{d-2}, \ldots , \sqrt{2} x_{d-1} x_{1}, \ldots , \sqrt{2} x_{2} x_{1}, \sqrt{2c} x_ d, \ldots , \sqrt{2c} x_1, c \rangle
$$
The polynomial kernel between two matrices $X$ and $Y$:
$$
K(x, y) = (\langle x, y \rangle + c)^p
$$
where $c$ is a scalar coefficient to trade off high-order and low-order terms, and $p$ is the degree of the polynomial kernel.

#### An example of Gaussian RBF Kernel

The Gaussian RBF kernel between $X$ and $Y$:
$$
K(x, y) = exp(-\gamma ||x-y||^2)
$$
where the squared pairwise Euclidean distances can be written into a simpler form
$$
\begin{aligned}
||x-y||^2 &= (x_1-y_1)^2 + (x_2-y_2)^2 + \ldots + (x_n-y_n)^2 \\ &= x_1^2+y_1^2-2x_1y_1 + \ldots + x_n^2+y_n^2-2x_ny_n \\&= x \cdot x + y \cdot y – 2x \cdot y\end{aligned}
$$
If the matrix $X$ is of size $N \times M$, and the matrix $Y$ is of size $K \times M$, then the distance matrix $D$ will be $N \times K$. The value of the entry in the $i$th row and $j$th column of $D$, is the distance between the $i$th row vector in $X$ and $j$th vector in $Y$.



# Additional Readings

About softmax function:

http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

About PCA:
https://online.stat.psu.edu/stat505/lesson/11