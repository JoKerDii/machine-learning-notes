# Selected Topics:

There are 19 topics and 10 exercises.

## 1. Dot Products and Norm

Dot product of vectors $a$ and $b$:
$$
\displaystyle  a \cdot b = a_1b_1+a_2b_2+\cdots +a_ nb_ n\qquad \text {where }\,  a= \begin{bmatrix}  a_1\\ a_2 \\ \vdots \\ a_ n \end{bmatrix} \text { and } b= \begin{bmatrix}  b_1\\ b_2 \\ \vdots \\ b_ n \end{bmatrix}.
$$

When considering $a$ and $b$ as vectors in $n$-dimensional space:
$$
\displaystyle a \cdot b = \| a\| \| b\| \cos \alpha
$$

where $\| \cdot\| $ refers to the length or known as **norm**.

$$
\displaystyle  \| a\|  = \sqrt{a_1^2+a_2^2+\cdots +a_ n^2}.
$$

## 2. Dot Products and Orthogonality

Two vectors $a$ and $b$ are **orthogonal** if the dot products of them equals to zero (the angle between them is $\pi/2$).

## 3. Unit Vectors

A **unit vector** is a vector with length 1. Given any vector $x$, the unit vector pointing in the same direction as $x$ is
$$
\frac{x}{\|x\|}
$$

## 4. Projection

The **projection** of $h$ onto $u$ is $|s|$:
$$
|s| = h \cdot u
$$
where $u$ is a unit vector in the direction of $s$.

> #### **Exercise 1:**
>
> Given 3-dimensional vectors $x^{(1)} = \begin{bmatrix} a_1\\ a_2\\ a_3\end{bmatrix}$ and $x^{(2)} = \begin{bmatrix} a_1\\ -a_2\\ a_3\end{bmatrix}$, which vector is in the same direction as the projection of $x^{(1)}$ onto $x^{(2)}$? What is the signed magnitude $c$ of the projection $p_{x^{(1)}\rightarrow x^{(2)}}$ of $x^{(1)}$ onto $x^{(2)}$? More precisely, let $u$ be the unit vector in the direction of the correct choice above, find a number $c$ such that $p_{x^{(1)}\rightarrow x^{(2)}}=cu$.
>
> > **Answer**:
> >
> > 1\) In the same direction of $x^{(2)}$; 2) $c = \frac{(a_1^2 - a_2^2 + a_3^2)}{\sqrt{(a_1^2 + a_2^2 + a_3^2)}}$
>
> >**Solution**:
> >
> >2\) The project has magnitude $c = \|x^{(1)}\|cos\alpha$. As we know $cos\alpha = \frac{x^{(1)}x^{(2)}}{\|x^{(1)}\|\|x^{(2)}\|}$. The projection thus has magnitude $c = \frac{x^{(1)} \cdot x^{(2)}}{\| x^{(2)}\| }$. By plugging in $x^{(1)}$ and $x^{(2)}$ we get the result.
> >
> >We can take a further step to find the final vector projection: we scale the unit vector $u =\frac{x^{(2)}}{\| x^{(2)}\| }$ in the direction of vector projection by the magnitude/length $c = \| p_{x^{(1)}\rightarrow x^{(2)}}\|$. So the projection $p_{x^{(1)}\rightarrow x^{(2)}}$ is
> >$$
> >p_{x^{(1)}\rightarrow x^{(2)}} = cu =\| p_{x^{(1)}\rightarrow x^{(2)}}\| \frac{x^{(2)}}{\| x^{(2)}\| }.
> >$$

## 5. Representation of Planes

A **hyperplane** in $n$ dimensions is a $n−1$ dimensional subspace. In general, a hyperplane in $n$-dimensional space can be represented as $\theta _0 + \theta _1 x_1 + \theta _2 x_2 + \cdots + \theta _ n x_ n = 0$. This representation can be defined by an $n$-dimensional vector $\theta = \begin{bmatrix} \theta _1 \\ \theta _2 \\ \vdots \\ \theta _ n \end{bmatrix}$ and offset $\theta_0$. 

One feature of this representation is that the vector $\theta$ is **normal** to the plane. Thus the $n$-dimensional vector $\theta = \begin{bmatrix} \theta _1 \\ \theta _2 \\ \vdots \\ \theta _ n \end{bmatrix}$  is  a normal vector for the plane.

## 6. Orthogonality Check

A vector $x$ is **orthogonal** to the plane if and only if it is collinear with the normal vector $\theta$ of the plane. (Conversely, any vector **parallel** to a plane must be orthogonal to the normal vector).

Check the orthogonality by checking whether:
$$
x = \alpha \theta \text{ for some } \alpha \in R
$$
Check the parallelity by checking whether：
$$
x \cdot \theta = 0
$$

>#### **Exercise 2:**
>
>Find an expression for the **orthogonal projection** of a point $v$ onto a plane $P$ that is characterized by $\theta$ and $\theta_0$. 
>
>> **Answer**:
>> $$
>> v - \frac{(\frac{(v \cdot \theta) + \theta_0}{\| \theta \|}) \cdot \theta}{ \| \theta \|}
>> $$
>
>> **Solution**:
>> Since $v−x$ is collinear with the normal, $v−x=\lambda \theta$ for some $\lambda$. Also, $x$ lies in the plane, so $\theta \cdot x + \theta_0 = 0$. 
>> $$
>> \displaystyle  (v - \lambda \theta )\cdot \theta + \theta _0 = 0
>> $$
>> Solve this to get the value of $\lambda$ and plug it back to find the orthogonal projection:
>> $$
>> x = \displaystyle  v-\frac{v\cdot \theta + \theta _0}{\| \theta \| }\hat{\theta }
>> $$



## 7. Perpendicular Distance to Plane

The distance from a point to a plane is the distance between the point $p$ and the projection of point $p$ onto the plane $q$.

**Approach of computing the distance**:  find any point $r$ on the plane, the distance equals to the projection of the vector $h$ from $r$ to point $p$ in the direction of the normal vector $s$ of the plane.
$$
|s| = |h \cdot u|
$$

>#### **Exercise 3:**
>
>Given a point x in $n$-dimensional space and a hyperplane described by $\theta$ and $\theta_0$, find the **signed distance** between the hyperplane and $x$. This is equal to the perpendicular distance between the hyperplane and $x$, and is positive when $x$ is on the same side of the plane as $\theta$ points and negative when $x$ is on the opposite side.
>
>> **Answer**:
>> $$
>> d_x = \frac{\theta \cdot x + \theta_0}{ \|\theta\|}
>> $$
>
>> **Solution**:
>> Define a plane: $\theta_1 \cdot x + \theta_2 \cdot y + \theta_0 = 0$, and a point $p$ outside of the plane $(x_1, y_1)$.
>> Find a random point $r$ on the plane: $(x_0, y_0)$.
>> So the vector $h$ from $r$ to $p$ is : $h = p-r = [x_1 - x_0, y_1 - y_0]$.
>> The projection from $h$ onto the normal vector $s$ is 
>> $$
>> Proj_{h\rightarrow s} = \frac{h \cdot s}{\|s\|} = \frac{[x_1 - x_0, y_1 - y_0] \cdot [\theta_1, \theta_2]}{\sqrt{\theta_1^2 + \theta_2^2}} = \frac{\theta_1 x_1 + \theta_2 y_1 - \theta_1 x_0 - \theta_2 y_0}{\sqrt{\theta_1^2 + \theta_2^2}}
>> $$
>> Since $\theta_1 x_0 + \theta_2 y_0 = -\theta_0$, the projection / perpendicular distance between point $p$ and the plane is:
>> $$
>> Proj_{h\rightarrow s} = \frac{\theta_1 x_1 + \theta_2 y_1 + \theta_0}{\sqrt{\theta_1^2 + \theta_2^2}}
>> $$
>> In general, it can be written as
>> $$
>> d_x = \frac{\theta \cdot x + \theta_0}{ \|\theta\|}
>> $$

## 8. Univariate Gaussians

A univariate **Gaussian** or **normal distributions** can be completely determined by its mean and variance.

Gaussian distributions can be applied to a large numbers of problems because of the **central limit theorem (CLT)**. The CLT posits that when a large number of **independent and identically distributed ((i.i.d.)** random variables are added, the **cumulative distribution function (cdf)** of their sum is approximated by the cdf of a normal distribution.

Recall the **probability density function (pdf)** of the univariate Gaussian with mean $\mu$ and variance $\sigma^2$, $\mathcal{N}(\mu , \sigma ^2)$.
$$
\displaystyle f_ X(x) = \frac{1}{\sqrt{2\pi \sigma ^2}} e^{-(x - \mu )^2/(2\sigma ^2)}.
$$

> #### **Exercise 4:**
>
> Let $\displaystyle X\sim \mathcal{N}\left(\mu ,\sigma ^2\right)$, $Y = 2X$. Write down the pdf of the random variable $Y$.
>
> > **Answer**:
> > $$
> > \displaystyle f_Y(y) = \frac{1}{2\sigma\sqrt{2\pi}} e^{-(x - 2\mu )^2/(8\sigma ^2)}.
> > $$
>
> > **Solution 1:**
> >
> > $Y=2X \sim \mathcal{N}\left(2\mu , 4\sigma ^2\right)$ since:
> > $$
> > E[2X] = 2E[X]\\Var[2X] = 2^2 Var[X] = 4Var[X]
> > $$
> > Therefore,
> > $$
> > \displaystyle f_Y(y) = \frac{1}{2\sigma\sqrt{2\pi}} e^{-(x - 2\mu )^2/(2(4\sigma ^2))}.
> > $$
>
> > **Solution 2:**
> >
> > In general, for any continuous random variable $X$ and any continuous monotonous function $g$, such that $Y = g(X)$, the pdf of  $Y$ is given by
> > $$
> > \displaystyle f_Y(y)=\frac{f_ X(x)}{|g'(x)|}\qquad \text {where } x=g^{-1}(y).
> > $$
> > In this problem, $\displaystyle X\sim \mathcal{N}\left(\mu ,\sigma ^2\right), Y = g(X) = 2X$, and $g'(X) = 2$. Therefore,
> > $$
> > \displaystyle f_Y(y)=\frac{f_ X\left(\frac{y}{2}\right)}{\left|g'\left(\frac{y}{2}\right)\right|} =\frac{1}{2 \sigma \sqrt{2\pi }} \exp \left(-\frac{\left((y-2\mu )\right)^2}{2(4)\sigma ^2}\right)
> > $$

## 9. Quantiles

The **quantile** of order $1−\alpha$ of a variable $X$, denoted by $q_{\alpha}$ (specific to a particular $X$), is the number such that $\displaystyle \mathbf{P}\left(X\leq q_{\alpha }\right)=1-\alpha$. 

Thus the area of shaded region in one tail is $\alpha$, and the area of shaded regions in two tail is $2 \alpha$.

## 10. 1D Optimization via Calculus

Given a function $f(x)$ defined on the interval $[a, b]$, the **critical points** of $f$ are those $x∈R$ such that $f′(x)=0$.

The first derivative of $f$ gives critical points where the derivatives equal to zero. The second derivative of $f$ gives information about whether the curve is concave or convex at the critical points, or whether the critical points are local minimum or maximum.

To figure out the global extrema, we need to test the critical points as well as the endpoints at $a$ and $b$.

## 11. Strict Concavity

A twice-differentiable function $f:I→R$, where $I$ is a subset of $R$, is **strictly concave** if $f''(x)<0$ for all $x∈I$.

> #### **Exercise 5:**
>
> Which of the following functions are strictly concave? (Choose all that apply.)
>
> A. $f_1(x) = x$ on $R$
>
> B. $f_2(x) = -e^{-x}$ on $R$
>
> C. $f_3(x) = x^{0.99}$ on the interval $(0, \infty)$
>
> D. $f_4(x) = x^2$ on $R$ 
>
> > **Answer**: BC
>
> > **Solution**:
> >
> > A. $f_1(x)=x$ is **not** strictly concave because $f_1''(x)=0$.
> >
> > B. $f_2(x)=−e^{−x}$ is strictly concave because $f_2''(x)=−e^{−x}<0$ for all $x∈R$.
> >
> > C. $f_3(x)=x^{0.99}$ is strictly concave because $f_3''(x)=(0.99)(−.01)x^{−1.01}<0$ for all $x∈(0,\infty)$.
> >
> > D. $f_4(x)=x^2$ is **not** strictly concave because $f_4''(x)=2>0$. In fact, this function is strictly **convex**.

## 12. Multivariable Calculus Review: Simple Gradient

Let
$$
f: \displaystyle  \mathbb {R}^ d \to R\\
\theta =\begin{pmatrix} \theta _1\\ \theta _2\\ \vdots \\ \theta _ d\end{pmatrix} \to f(\theta)
$$
denote a **differentiable** function. The gradient of $f$ is the vector-valued function
$$
{\nabla _\theta }  f: R^d \to R^d\\
\theta =\begin{pmatrix} \theta _1\\ \theta _2\\ \vdots \\ \theta _ d\end{pmatrix} \to \left.\begin{pmatrix}  \frac{\partial f }{\partial \theta _1}\\ \frac{\partial f }{\partial \theta _2}\\ \vdots \\ \frac{\partial f }{\partial \theta _ d}\end{pmatrix}\right|_{\theta }.
$$

> #### **Exercise 6:**
>
> Consider $f(\theta) = \theta_1^2 + \theta_2^2$. 
>
> 1\) Compute the gradient $\nabla f$.
>
> 2\) What's the shape of $f$ ?
>
> 3\) Does the graph (surface) of $f(\theta)$ have a global maximum, or global minimum, or neither?
>
> 4\) At each point $\theta=(\theta_1,\theta_2)$ in the $(\theta_1,\theta_2)$-plane, $f(\theta)$ decreases in which direction?
>
> > **Answer**:
> >
> > 1\) $[2\theta_1, 2\theta_2]$
> >
> > 2\) paraboloid, $f(\theta)$ is a circle at each $f(\theta) = K$.
> >
> > 3\) global minimum
> >
> > 4\) $- \nabla _\theta f(\theta )$
>
> > **Solution**:
> >
> > The graph of $f(\theta)$ is a paraboloid that opens downwards. Its global minimum is at $\theta = (0,0)$. Since $\nabla _\theta f(\theta )=(2\theta _1,2\theta _2)^ T$, $- \nabla _\theta f(\theta )$ points towards the origin at all points $\theta$.

## 13. Gradient Ascent or Descent

**Gradient ascent/descent** methods are typical tools for maximizing/minimizing functions. Consider the function $L(x,\theta)$ where $\theta =[\theta _1,\theta _2,\ldots ,\theta _ n]^ T$ and $x=[x_1,x_2,\ldots ,x_ n]^ T$. Our goal is to select $\theta$ such to maximize/minimize the value of $L$ while keeping $x$ fixed. This selection is a consecutive update procedure that will hopefully eventually converge to the global minimum of $L(x,\theta)$ (if it exists).

When treating $x$ as a constant and differentiating w.r.t $\theta$:
$$
\displaystyle \nabla _{\theta }L(x,\theta )=\begin{pmatrix}  \frac{\partial }{\partial \theta _1}L(x,\theta )\\ \vdots \\ \frac{\partial }{\partial \theta _ n}L(x,\theta )\end{pmatrix}.
$$

> #### **Exercise 7:**
>
> If $\theta ' = \theta +\epsilon \cdot \nabla _{\theta }L(x,\theta )$,
>
> where $\epsilon$ is a small positive real number, which of the following is true?
>
> A. $L(x, \theta ')>L(x, \theta )$.
>
> B. $L(x, \theta ')<L(x, \theta )$.
>
> > **Answer**: A
>
> > **Solution**:
> >
> > If the gradient is positive,  by adding $\epsilon \cdot \nabla _{\theta }L(x,\theta )$ to $\theta$, we move it toward a positive direction. This increases $L(x, \theta)$. If the gradient is negative, we move it toward a negative direction, which again increases $L(x, \theta)$.
> >
> > Alternatively, to minimize $L(x, \theta)$, the updating rule should be $\theta' = \theta - \epsilon \cdot \nabla _{\theta }L(x,\theta )$.
> >
> > There are more complications in higher dimensions, but this is the basic idea behind **stochastic gradient descent**, which forms the backbone of modern machine learning

## 14. Vector Inner and Outer product

The product $u^Tv$ evaluates the **inner product** (also called the **dot product** ) of $u$ and $v$.

The inner product of $u$ and $v$ is sometimes written as $⟨u,v⟩$.

The product $uv^T$ evaluates the **outer product** of $u$ and $v$.

## 15. Linear Independence, Subspaces and Dimension

Vectors $\mathbf v_1, \ldots , \mathbf v_ n$ are **linearly dependent** if there exist scalars $c_1, \ldots , c_ n$ such that:

(1) not all $c_i's$ are zero, and 

(2) $c_1 \mathbf v_1 + \cdots + c_ n \mathbf v_ n = 0$

Otherwise, they are **linearly independent**: the only scalars $c_1, \ldots , c_ n$ that satisfy $c_1 \mathbf v_1 + \cdots + c_ n \mathbf v_ n = 0$ are $c_1 = \cdots = c_ n = 0$.

The collection of non-zero vectors $\mathbf v_1, \ldots , \mathbf v_ n \in \mathbb {R}^ m$ determines a **subspace** of $R^m$, which is the set of all linear combinations $c_1 \mathbf v_1 + \cdots + c_ n \mathbf v_ n$ over different choices of $c_1,\ldots ,c_ n \in \mathbb {R}$. 

The **dimension** of this subspace is the size of the **largest possible, linearly independent** sub-collection of the (non-zero) vectors $\mathbf v_1, \ldots , \mathbf v_ n$.

> #### **Exercise 8:**
>
> what is the largest possible rank of an $m×n$ matrix?
>
> > **Answer**: $min(m,n)$
>
> > **Solution**: Since  rank = column rank = row rank.

## 16. Invertibility of a matrix

An $n×n$ matrix $A$ is **invertible** if and only if $A$ has **full rank**, i.e. $rank(A)=n$.

We can obtain the reduced upper triangular matrix by Gaussian Elimination. In general, the upper triangular matrix with nonzero entries along the diagonal has full rank.

## 17. Determinant

The **determinant** $det(A)$ of a square matrix $A$ indicates whether it is invertible. For $2×2$ matrices, it has the formula
$$
\mathrm{det}\left( \begin{array}{cc} a &  b \\ c &  d \end{array} \right) = ad-bc.
$$
A useful property holds for all matrices:
$$
\mathrm{det}(\mathbf{A}) = \mathrm{det}\mathbf{A}^ T.
$$

## 18. Quadratic Polynomials

Recall a **degree** n polynomial in $x_1,x_2,\ldots , x_ k$ are all linear combinations of monomials in $x_1,x_2,\ldots , x_ k$, where **monomials** in $x_1,x_2,\ldots , x_ k$ are **unordered words** using $x_1,x_2,\ldots , x_ k$ as the letters.

> #### Example 1:
>
> A degree 2 (quadratic) polynomial in 1 variable $x$: $ax^2+b x+ c$ ,which has three coefficients $a, b, c$. 
>
> In linear algebra, the space of degree 2 polynomials in 1 variable is of dimension 3 since it consists of all linear combinations of 3 linearly independent vectors $x^2,x$, and 1.

> #### Example 2:
>
> A degree 2 polynomials in 2 variables $x_1, x_2$: $ax_1^2+b x_2^2+ c x_1x_2+d x_1+ e x_2 +f$, which has six coefficients. 
>
> In linear algebra, the space of degree 2 polynomials in 2 variables is of dimension 6 since it consists of all linear combinations of 6 linearly independent vectors $x_1^2,\,  x_2^2,\,  x_1x_2,\, x_1,\,  x_2, \,$ and 1.

> #### **Exercise 9:**
>
> Consider degree 2 polynomials in 3 variables $x_1,x_2,x_3$. 
>
> 1\) How many coefficients are needed to completely determine such a polynomial? Equivalently, what is the dimension of the space of polynomials in 3 variables such polynomials? 
>
> 2\) What is dimension of the polynomials of degree $N$ in $K$ variables?
>
> > **Answer**: 1\) 10; 2\)  $\displaystyle \binom {K}{N} + NK+1$
>
> > **Solution**: 
> >
> > We count the number of monomials of length 2,1,0:
> >
> > - The monomials of length 2 are unordered pairs of $x_1,x_2,\, x_3$, hence there are $\binom {3}{2}$ This list consists of $x_1^2, x_2^2,x_3^2, x_1x_2,x_1x_3,x_2x_3$. [In general, $\binom {K}{N} + \sum_{i=1}^{N-1} K$]
> > - The monomials of length 1 are $x_1, \, x_2,\, x_3$. [In general, $K$]
> > - The monomial of length 0 is the constant term, i.e. 1. [In general, 1]
> >
> > In general, the dimension of the polynomial of degree $N$ in $K$ variables is: 
> > $$
> > \binom {K}{N} + \sum_{i=1}^{N} K + 1 = \binom {K}{N} + N K + 1
> > $$



## 19. Eigenvalues, Eigenvectors and Determinants

For general $n×n$ matrices:

The **product of the eigenvalues** is always equal to the **determinant**.

The **sum of the eigenvalues** is always equal to the **trace** of the matrix.

> #### **Exercise 10:**
>
> If a (nonzero) vector is in the nullspace of a square matrix $A$, is it an eigenvector of $A$? Answer the question and choose equivalent statement below (Choose all that apply).
>
> A. There exists a nonzero solution to $Av = 0$
>
> B. $det(A) = 0$
>
> C. $det(A) \neq 0$ 
>
> D. $NS(A) = 0$
>
> E. $NS(A) \neq 0$
>
> > **Answer**: Yes ; ABE
>
> > **Solution**: 
> >
> > If a vector $v$ is in the nullspace of $A$, then $Av=0=(0)v$. So it is an eigenvector of $A$ associated to the eigenvalue 0.
> >
> > If 0 is an eigenvalue for a matrix $A$, then by definition, there exists a nonzero solution to $Av=0$ (linearly dependent); that is, $NS(A)≠0$, and this only happens if and only if $det(A)=0$.



# Additional Readings

http://www.math.lsa.umich.edu/~glarose/classes/calcIII/

http://faculty.bard.edu/belk/math213

http://cs229.stanford.edu/section/cs229-linalg.pdf

http://cs229.stanford.edu/section/cs229-prob.pdf