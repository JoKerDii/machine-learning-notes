# Problem 2

There are 3 questions.

## Linear Support Vector Machines

For this problem we aim to minimize the training objective for a Support Vector Machine (with margin loss), which can be seen as optimizing a balance between the average hinge loss over the examples and a regularization term that tries to keep the parameters small (larger margin). The balance is set by the regularization parameter $\lambda > 0$. Here we assume no offset parameter $\theta_0$.

The training objective:
$$
\displaystyle  \left[\frac{1}{n}\sum _{i=1}^{n}Loss_ h (y^{(i)}\, \theta \cdot x^{(i)}\, )\right] + \frac{\lambda }{2}{\left\|  \theta  \right\| ^{2}} = \frac{1}{n}\sum _{i=1}^{n}\left[Loss_ h (y^{(i)}\, \theta \cdot x^{(i)}\, ) + \frac{\lambda }{2}{\left\|  \theta  \right\| ^{2}}\right]
$$
where the hinge loss is 
$$
\text {Loss}_ h(y(\theta \cdot x)) = \max \{ 0,1-y(\theta \cdot x)\}
$$
The optimization problem is defined as
$$
\displaystyle  \hat{\theta } = \text {argmin}_\theta [ \text {Loss}_ h (y\, \theta \cdot x\, ) + \frac{\lambda }{2}{\left\|  \theta  \right\| ^{2}}]
$$

> #### Question 6: 
>
> In this question, suppose that $\text {Loss}_ h (y(\hat{\theta }\cdot x))>0$. Under this hypothesis, solve for optimization problem and express $\hat{\theta}$ in terms of $x, y$ and $\lambda$.
>
> > **Answer**: $\frac{x \cdot y}{\lambda}$
>
> > **Solution**: Solve the minimization problem by solving:
> > $$
> > 0 = \nabla _\theta [\text {Loss}_ h (y(\theta \cdot x))] + \nabla _\theta [\frac{\lambda }{2} \left\|  \theta  \right\| ^2]
> > $$
> > Given $\displaystyle  \text {Loss}_ h (y(\hat{\theta }\cdot x)) > 0$, we have 
> > $$
> > \begin{aligned}
> > \displaystyle  \text {Loss}_ h (y(\hat{\theta }\cdot x)) & = 1-y(\theta \cdot x)\\
> > \displaystyle \nabla _\theta [\text {Loss}_ h (y(\theta \cdot x))] & = -yx
> > \end{aligned}
> > $$
> > Plug this back in the previous equation, we get
> > $$
> > \begin{aligned}
> > 0 & = \lambda \hat{\theta} - yx\\
> > \hat{\theta } & = \frac{1}{\lambda } y x
> > \end{aligned}
> > $$

> #### Question 7: 
>
> Given the following numerical examples (points lie on a two dimensional space)
> $$
> \lambda = 0.5,~ y = 1,~ x = \begin{bmatrix} 1 \\ 0\end{bmatrix}
> $$
> Let $\hat{\theta} = [\hat{\theta_1}, \hat{\theta_2}]$, solve for $\hat{\theta_1}, \hat{\theta_2}$.
>
> > **Answer**: $\hat{\theta_1} = 1$, $\hat{\theta_2} = 0$.
>
> > **Solution**: 
> >
> > Given $Loss_ h (y(\theta \cdot x)) \le 0$, which implies that $y(\theta \cdot x) \ge 1$., we are left with minimizing $\frac{\lambda }{2}\left\|  \theta  \right\| ^2$ under the constraint $y(\theta \cdot x) \ge 1$.
> >
> > The geometry of the problem implies that $y(\theta \cdot x) = 1$, which is $1 - (\hat{\theta _1}* 1 + \hat{\theta _2}* 0) = 0$. We get $\hat{\theta _1} = 1$. Then, to minimize $\| \theta\|, \hat{\theta_2} = 0$.
> >
> > Therefore $\hat{\theta } = \begin{bmatrix} 1 \\ 0\end{bmatrix}$.
> >
> > In fact $\hat{\theta}=\frac{x}{y \|x\|^2}$, the solution of the optimization is necessarily of the form $\hat{\theta } = \eta y x$ for some read $\eta > 0$.

> #### Question 8:
>
> Let $\hat{\theta} = \hat{\theta}(\lambda)$ be the solution as a function of $\lambda$.
>
> For what value of $\|x\|^2$, the training example $(x,y)$ will be misclassified by $\hat{\theta}(\lambda)$?
>
> > **Answer**: $\|x\|^2 = 0$.
>
> > **Solution**: For a misclassified point
> > $$
> > y\hat{\theta }\cdot x \leq 0
> > $$
> > The hinge loss is greater than 0. Thus the solution of minimization problem is 
> > $$
> > \hat{\theta }=\frac{y x}{\lambda }
> > $$
> > Plug this in the inequation, we get
> > $$
> > y\hat{\theta }\cdot x = \frac{y^{2} \left\|  x \right\| ^{2}}{\lambda }{\leq }0
> > $$
> > As all terms are non-negative, making it impossible to be < 0. But if $\|x\| = 0$, the product can be 0.



