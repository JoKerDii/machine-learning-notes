# Problem 3

## Ridge Regression

Consider fitting a $l_2$-regularized linear regression model to data $(x^{(1)},y^{(1)}),\dots ,(x^{(n)}, y^{(n)})$ where $x^{(t)}, y^{(t)} \in \mathbb {R}$ are scalar values for each $t = 1, ..., n$. To fit the parameters of this model, one solves:
$$
\min _{\theta \in \mathbb {R}, \  \theta _0 \in \mathbb {R}} L(\theta , \theta _0)
$$
where
$$
L(\theta , \theta _0) = \sum _{t=1}^ n (y^{(t)} - \theta x^{(t)} - \theta _0)^2 \  \  + \  \lambda \theta ^2
$$
Note that the objective function is **convex**, so any point where $\nabla L (\theta_0, \theta) = 0$ is a global minimum. The expression for the gradient of the objective function in terms of $\theta$ and $\theta_0$:
$$
\begin{aligned}
\nabla L & = \Big[\frac{\partial L}{\partial \theta _0} , \frac{\partial L}{\partial \theta } \Big] = 0 \\
\frac{\partial L}{\partial \theta _0} & = -2 \sum _{t=1}^ n (y^{(t)} -\theta x^{(t)} - \theta _0) \\
\frac{\partial L}{\partial \theta } & = 2\lambda \theta - 2 \sum _{t=1}^ n (y^{(t)} -\theta x^{(t)} - \theta _0) x^{(t)}
\end{aligned}
$$
To solve the ridge regression minimization, the closed from expression for $\theta_0$ and $\theta$ can be computed by first fixing $\theta$ and writing down the expression for the optimal $\hat{\theta_0}$, then plugging in $\hat{\theta_0}$ to compute the optimal  $\hat{\theta}$.
$$
\displaystyle  \frac{\partial }{\partial \theta _0} = - 2 \sum _{t=1}^ n (y^{(t)} -\theta x^{(t)} - \theta _0) = - 2 \sum _{t=1}^ n (y^{(t)} -\theta x^{(t)}) + 2 \sum _{t=1}^ n \theta _0 = 0 \\
\implies \displaystyle  -2n \theta _0 = -2\sum _{t=1}^ n (y^{(t)} - \theta x^{(t)}) \\  \implies \theta _0 = \frac{1}{n} \sum _{t=1}^ n (y^{(t)} - \theta x^{(t)}) \\
$$

$$
\displaystyle  \frac{\partial }{\partial \theta } = 2\lambda \theta - 2 \sum _{t=1}^ n (y^{(t)} -\theta x^{(t)} - \theta _0) x^{(t)} \\
\displaystyle  = 2\lambda \theta - 2 \sum _{t=1}^ n \Big(y^{(t)} -\theta x^{(t)} - \Big[\frac{1}{n} \sum _{s=1}^ n (y^{(s)} - \theta x^{(s)}) \Big] \Big) \cdot x^{(t)} = 0 \\ 
\implies \displaystyle  \lambda \theta - \sum _{t=1}^ n x^{(t)} y^{(t)} + \theta \sum _{t=1}^ n {x^{(t)}}^2 + \frac{1}{n} \sum _{t=1}^ n \sum _{s=1}^ n (y^{(s)} - \theta x^{(s)}) x^{(t)} = 0\\
\implies \displaystyle  \lambda \theta - \sum _{t=1}^ n x^{(t)} y^{(t)} + \theta \sum _{t=1}^ n {x^{(t)}}^2 + \frac{1}{n} \sum _{t=1}^ n \sum _{s=1}^ n y^{(s)} x^{(t)} - \frac{1}{n} \theta \sum _{t=1}^ n \sum _{s=1}^ n x^{(s)} x^{(t)} = 0 \\
\implies \displaystyle  \widehat{\theta } = \frac{\sum _{t=1}^ n x^{(t)} y^{(t)} - \frac{1}{n} \sum _{t=1}^ n \sum _{s=1}^ n y^{(s)} x^{(t)}}{\lambda + \sum _{t=1}^ n {x^{(t)}}^2 - \frac{1}{n} \sum _{t=1}^ n \sum _{s=1}^ n x^{(s)} x^{(t)} } \\
$$

Note that if we define $\displaystyle \bar{x} = \frac{1}{n} \sum _{t=1}^ n x^{(t)}$, then we can rewrite the above expression in a better form: 
$$
\widehat{\theta } = \frac{ \sum _{t=1}^ n (x^{(t)} - \bar{x}) y^{(t)} }{\lambda + \sum _{t=1}^ n x^{(t)}(x^{(t)} - \bar{x}) }
$$
Finally, we can plug $\hat{\theta}$ back into expression of $\hat{\theta_0}$ to find the corresponding $\hat{\theta_0}$.