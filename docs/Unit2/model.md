# Implementation Ideas

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
\begin{align}
\frac{\partial J(\theta )}{\partial \theta _ m} & = \frac{\partial }{\partial \theta _ m}\Bigg[-\frac{1}{n}\Bigg[\sum _{i=1}^ n \sum _{j=0}^{k-1} [[y^{(i)} == j]] \log p(y^{(i)} = j | x^{(i)}, \theta ) \Bigg] + \frac{\lambda }{2}\sum _{j=0}^{k-1}\sum _{i=0}^{d-1} \theta _{ji}^2\Bigg] \\
& = -\frac{1}{\tau n} \sum _{i = 1} ^{n} [x^{(i)}([[y^{(i)} == m]] - p(y^{(i)} = m | x^{(i)}, \theta ))] + \lambda \theta _ m
\end{align}
$$
To run the gradient descent, we will update $\theta$ at each step with $\theta \leftarrow \theta - \alpha \nabla _{\theta } J(\theta )$, where $\alpha$ is the learning rate.





# Additional Readings

About softmax function:

http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

