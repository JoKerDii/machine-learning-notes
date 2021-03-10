# Problem 1

There are 2 questions

## Feature Vectors Transformation

Consider a sequence of $n$-dimensional data points, $x^{(1)},x^{(2)},...,$ and a sequence of $m$-dimensional feature vectors, $z^{(1)},z^{(2)},...,$ extracted from the $x$'s by a linear transformation, $z^{(i)}=Ax^{(i)}$. If $m$ is much smaller than $n$, you might expect that it would be easier to learn in the lower dimensional feature space than in the original data space.

> #### Question 1:
>
> Suppose $n=6, m=2, z_1$ is the average of the elements of $x$, and $z_2$ is the average of the first three elements of $x$ minus the average of fourth through sixth elements of $x$. Determine $A$.
>
> > **Answer**: $A = \begin{bmatrix}  1/6~~~~1/6~~~~1/6~~~~1/6~~~~1/6~~~~1/6\\ 1/3~~~1/3~~~1/3~-1/3~-1/3~-1/3\\\end{bmatrix}$
>
> > **Solution**: 
> >
> > $z^{(i)}=Ax^{(i)}$ where in this question $i \in \{1,2\}$, $z$ is $2 \times 2$, $A$ is $2 \times 6$, $x$ is $6 \times 2$.
> > $$
> > z_1 = [a_{11},a_{12},a_{13},a_{14},a_{15},a_{16}] \cdot x^{(i)}\\
> > z_2 = [a_{21},a_{22},a_{23},a_{24},a_{25},a_{26}] \cdot x^{(i)}
> > $$

> #### Question 2:
>
> Using the same relationship between $z$ and $x$ as defined above, suppose $h(z)=sign(θ_z⋅z)$ is a linear classifier for the feature vectors, and $g(x)=sign(θ_x⋅x)$ is a linear classifier for the original data vectors. Given a $θ_z$ that produces good classifications of the feature vectors, determine a $θ_x$ (an $n \times 1$ vector )that will identically classify the associated $x$'s.
>
> > **Answer**: $\theta _ x = A^\top \theta _ z$.
>
> > **Solution**: From above, we have the relationship that $z=Ax$. Therefore $θ_z⋅z=θ_z⋅Ax=θ^⊤_zAx=(A^⊤θ_z)⋅x$. So take $θ_x=A^⊤θ_z$ and we have the same classifier in original space.

#### Some key points:

1. Given the same classifiers as in question 2, assume the matrix $A$ is **fixed**, if there is a $\theta_x$ that produces good classifications of the data vectors, there is **not always** be a $\theta_z$ that will identically classify the associated $z$'s

There is a formal condition when there will be a $\theta_z$ that will identically classify the associated $z$'s. Formally, given a $\theta_x$ that correctly classifies the points in data space of dimension $m < n$, we are looking for $\theta_z$ such that $\theta _ x^{T}x = \theta _ z^{T}Ax$ for all $x$. Find such $\theta_z$ is equivalent to solving the overdetermined linear system $A^{T}\theta _ z = \theta _ x$, which can be down only if the system is consistent, i.e. if it has solution. This will happen if and only if $\theta_x$ is in the span of the columns of $A^T$.

By looking at the equivalent system $AA^{T}\theta _ z = A \theta _ x$ we can identify two cases:

a) $A$ **has linearly independent rows.** In this case $AA^T$ is invertible, so there is a unique solution given by $\theta _ z = (AA^{T})^{-1}A \theta _ x$.

b) $A$ **has linearly dependent rows.** In this case, the system is indeterminate and has an infinite number of solutions.

The matrix $(AA^{T})^{-1}A$ of part $(i)$ is known as the **Moore-Penrose pseudo-inverse** of $A^T$, and it is denoted by $(A^ T)^{\dagger }$.

2. Given the conditions in the first point above, assume this time we have a $m \times n$ matrix $A$ which is changeable. In this case, there is **always** be a $\theta_z$ that will identically classify the associated $z$'s.

Given flexibility in both $A$ and $θ_z$, we are looking for $A$, $θ_z$ such that $A^⊤θ_z=θ_x$. We can achieve this by simply setting $θ_z=1$, the first row of $A$ to be $θ_x$, and the remaining rows to be 0:
$$
A^\top \theta _ z = \begin{bmatrix}  | &  |\\ \theta _ x &  0 \\ | &  | \end{bmatrix} \begin{bmatrix}  1 \\ 0 \end{bmatrix} = \theta _ x
$$

3. If $m < n$, we cannot find a more accurate classifier by training in $z$-space, as measured on the training data. If we measure on the unseen data, whether we could find a more accurate classifier depends on the arbitrary unseen data.

The accuracy in $z$-space is always bounded by the $x$ space, as we can always construct a classifier in $x$ space that corresponds to a classifier in $z$ space.

As for measuring on the unseen data, we can always construct a dataset that favors the classifier produced in $z$ space. We can do the same thing to the classifier produced in $x$ space as well.

