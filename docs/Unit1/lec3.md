# Margin Boundaries, Hinge Loss, and Regularization

There are 3 topics and 1 exercise

## 1. Margin Boundary

The **Decision Boundary** is the set of points $x$ which satisfy
$$
\theta \cdot x + \theta _0=0.
$$
The **Margin Boundary** is the set of points $x$ which satisfy
$$
\theta \cdot x + \theta _0= \pm 1.
$$
Recall that the distance from a point $P(x_0)$ to a line $\theta x_0 + \theta_0 = 0$ is $\frac{|\theta x_0 + \theta_0|}{\|\theta\|}$, so the distance from the decision boundary to the margin boundary is $\frac{1}{\mid \mid \theta \mid \mid }$. As we increase $∣∣θ∣∣$, $\frac{1}{∣∣θ∣∣}$ decreases.

## 2. Hinge Loss

**Hinge Loss**:
$$
Loss_h(z)=
    \begin{cases}
      0, & \text{if }\ z \geq 1 \\
      1-z, & \text{if }\ z < 1
    \end{cases}
$$
where $z = y^{(i)}(\theta x^{(i)} + \theta_0)$.

> #### **Exercise 14:**
>
> From the information given above, what is the value of $y^{(i)}(\theta \cdot x^{(i)} + \theta _0)$ and Hinge Loss for
>
> 1\) a correctly classified point $(x^{(i)}, y^{(i)})$ on the margin boundary?
>
> 2\) a correctly classified point $(x^{(i)}, y^{(i)})$ in the middle the margin?
>
> 3\) a point $(x^{(i)}, y^{(i)})$ at the decision boundary?
>
> 4\) a misclassified point $(x^{(i)}, y^{(i)})$ on the margin boundary?
>
> 5\) a misclassified point $(x^{(i)}, y^{(i)})$ in the middle the margin?
>
> > **Answer**: 
> >
> > 1\) 1 ; 0
> >
> > 2\) 1/2 ; 1/2
> >
> > 3\) 0 ; 1
> >
> > 4\) -1 ; 2
> >
> > 5\) -1/2; 3/2
>
> > **Solution**: 
> >
> > 1\) $\frac{|\theta x_0 + \theta_0|}{\|\theta\|} = |\theta x_0 + \theta_0| = 1$, thus, $y^{(i)}(\theta \cdot x^{(i)} + \theta _0) =1$, $Loss_h(1) = 1-1 = 0$.
> >
> > 2\) $\frac{|\theta x_0 + \theta_0|}{\|\theta\|} = |\theta x_0 + \theta_0| = 1/2$, thus, $y^{(i)}(\theta \cdot x^{(i)} + \theta _0) =1/2$, $Loss_h(1/2) = 1-1/2 = 1/2$
> >
> > 3\) $\frac{|\theta x_0 + \theta_0|}{\|\theta\|} = |\theta x_0 + \theta_0| = 0$, thus, $y^{(i)}(\theta \cdot x^{(i)} + \theta _0) =0$, $Loss_h(0) = 1-0 = 1$
> >
> > 4\) $\frac{|\theta x_0 + \theta_0|}{\|\theta\|} = |\theta x_0 + \theta_0| = 1$, thus, $y^{(i)}(\theta \cdot x^{(i)} + \theta _0) = -1$, $Loss_h(-1) = 1-(-1) = 2$
> >
> > 5\) $\frac{|\theta x_0 + \theta_0|}{\|\theta\|} = |\theta x_0 + \theta_0| = 1/2$, thus, $y^{(i)}(\theta \cdot x^{(i)} + \theta _0) = -1/2$ , $Loss_h(-1/2) = 1-(-1/2) = 3/2$

## 3. Regularization and Objective

Regularization towards max margin:
$$
\max\frac{1}{\|\theta\|} = \min \|\theta\| = \min \frac{1}{2} \|\theta\|^2
$$
The **objective** function: (objective function = average loss (for minimizing error) + regularization (for maximizing margin))
$$
J(\theta, \theta_0) = \frac{1}{n} \sum^n_{i=1} Loss_h (y^{(i)}(\theta x^{(i)} + \theta_0)) + \frac{\lambda}{2} \|\theta\|^2
$$
If we have large $\lambda$, We put more importance on maximizing the margin than minimizing errors.

