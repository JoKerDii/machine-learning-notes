# Nonlinear Classifier

There are topics and questions



## 1. Higher Order Feature Vectors

We can use linear classifiers to make non-linear predictions. The easiest way to do this is to first map all the examples $x∈R^d$ to different feature vectors $ϕ(x)∈R^p$ where typically $p$ is much larger than $d$. We would then use a linear classifier on the new (higher dimensional) feature vectors. As a result, all the linear classifiers remain applicable in new higher dimension, yet produce non-linear classifiers in the original coordinates.



> #### Exercise 17
>
> Given the training examples with $x=[x^{(t)}_1,x^{(t)}_2]∈R^2$ below, where a boundary between the positively-labeled examples and the negatively-labeled examples is an ellipse, which of the following feature vector(s) $ϕ(x)$ will guarantee that the training set ${(ϕ(x^{(t)}),y^{(t)}),t=1,…,n}$ (where $y^{(t)}$ are the labels) are linearly separable?
> (Choose all that apply.)
>
> A. $\phi (x)=[x_1,x_2]^ T$
>
> B. $\phi (x)=[x_1,x_2,x_1x_2 ]^ T$
>
> C. $\phi (x)=[x_1,x_2,x_1^2+x_2^2]^ T$
>
> D. $\phi (x)=[x_1,x_2,x_1^2+2x_2^2]^ T$
>
> E. $\phi (x)=[x_1,\, x_2,\, x_1x_2,\, x_1^2,\, x_2^2]^ T$
>
> ![ex17](../assets/images/U2/lec6.svg)
>
> > **Answer**: E
>
> > **Solution**: 
> >
> > Since a possible boundary is an ellipse, and from geometry that the equation of any ellipse can be given as
> > $$
> > \theta _1 x_1+\theta _2 x_2+\theta _3x_1x_2+\theta _4x_1^2+\theta _5 x_2^2+ \theta _0\, =\, \theta \cdot [x_1,\, x_2,\, x_1x_2,\, x_1^2,\, x_2^2]^ T\, +\theta _0=\, 0,
> > $$
> > for some $\theta =[\theta _1,\theta _2,\theta _3,\theta _4,\theta _5],\,$ so the feature map is defined as $\phi (x)=[x_1,x_2,x_1x_2,x_1^2,x_2^2]^ T$. The linear decision boundary in the $\phi$-coordinates is 
> > $$
> > \theta \cdot \phi (x)+\theta _0 = 0 \Longleftrightarrow \theta \cdot [x_1,x_2,x_1x_2,x_1^2,x_2^2]^ T+\theta _0 = 0
> > $$
> > A.  $ϕ(x)=[x_1,x_2]^T$ maps x to $x$ itself in $\R^2$, so the training set remains the same and not linearly-separable.
> >
> > C/D. $\phi (x)=[x_1,x_2,x_1^2+x_2^2]^ T\,$ gives decision boundaries of the form as follows which are circles in the $(x_1,y_1)$
> > $$
> > \theta \cdot [x_1,x_2,x_1^2+x_2^2]^ T+\theta _0\, = \theta _1 x_1+\theta _2 x_2+\theta _3\left(x_1^2+x_2^2\right)+ \theta _0\, =\, 0
> > $$



## 2. Non-linear Classification

By mapping examples into a feature representation and performing linear classification in the new feature coordinates, we get a **nonlinear classifier**.

* **Example**: When $x$ is actually two dimensional, living in $\R^2$. Then if we include the second order of features, we will get $[x_1,x_2,x_1^2,x_2^2,x_1x_2]$, which is in five-dimensional space.

We can add more linearly independent features/ feature coordinates, and get more powerful feature representations, more powerful classifiers.

> #### Exercise 18
>
> Counting Dimensions of Feature Vectors



## 3. Kernels

Computing the inner product between two feature vectors can be cheap even if vectors are very high dimensional. Instead of explicitly constructing feature vectors $sign(\theta \cdot \phi(x) + \theta_0)$, we express the linear classifiers in terms of kernels. The kernel function can be represented by
$$
K(x,x') = \phi(x) \cdot \phi(x') = (x \cdot x') + (x \cdot x')^2 + (x \cdot x')^3 ... = (1 + x \cdot x')^p, p = 1,2,...
$$


> #### Exercise 19: 
>
> Assume we map $x$ and $x′∈R^2$ to feature vectors $ϕ(x)$ and $ϕ(x′)$ given by
> $$
> \phi(x) = \big [x_1, \, x_2,\,  {x_1}^2,\,  \sqrt{2}x_1x_2,\,  {x_2}^2 \big ]\\
> \phi(x') = \big [x_1^\prime ,\,  x_2^\prime ,\,  {x_1^\prime }^2,\,  \sqrt{2}x_1^\prime x_2^\prime ,\,  {x_2^\prime }^2 \big ].
> $$
> Which of the following equals the dot product $\phi (x) \cdot \phi (x')$
>
> A. $x \cdot x'$ 
>
> B. $x \cdot x' + (x \cdot x')^2$
>
> C. $(x \cdot x')^2$
>
> D. $2(x \cdot x')^2$
>
> E. None of the above
>
> > **Answer**: B
>
> > **Solution**: 
> >
> > Expand $\phi (x) \cdot \phi (x')$ to get
> > $$
> > \begin{aligned}
> > \phi (x) \cdot \phi (x^\prime ) & = \displaystyle {x_1}{x_1^\prime } + {x_2}{x_2^\prime } + {x_1}^2{x_1^\prime }^2 + 2{x_1}{x_1^\prime }{x_2}{x_2^\prime } + {x_2}^2{x_2^\prime }^2 \\
> > & = \left({x_1}{x_1^\prime } + {x_2}{x_2^\prime }\right)+ \left({x_1}{x_1^\prime } + {x_2}{x_2^\prime }\right)^2 \\
> > & = x \cdot x^\prime + (x \cdot x^\prime )^2
> > \end{aligned}
> > $$
> > Notice the coefficient $\sqrt{2}$ of the $x_1x_2$ terms is necessary for rewriting $ϕ(x)⋅ϕ(x′)$ as the function above of $x⋅x′$.

> #### Exercise 20
>
> Which of the following feature vectors $ϕ(x)$ produces the kernel
> $$
> K(x, x') \, =\,  \phi (x)\cdot \phi (x')\, =\, x_1x_1^\prime + x_2x_2^\prime + x_3x_3^\prime + x_2x_3^\prime + x_3x_2^\prime
> $$
> A. $\phi (x)=\big [x_1, x_2, x_3\big ]$
>
> B. $\phi (x)=\big [x_1 + x_2 + x_3\big ]$
>
> C. $\phi (x)=\big [x_1, x_2 + x_3\big ]$
>
> D. $\phi (x)=\big [x_1 + x_3, x_1 + x_2\big ]$
>
> > **Answer**: C
>
> > **Solution**: 
> > $$
> > [x_1, x_2 + x_3] \cdot [x_1', x_2' + x_3'] =\, x_1x_1^\prime + x_2x_2^\prime + x_3x_3^\prime + x_2x_3^\prime + x_3x_2^\prime
> > $$
> > The fact that there are mixed terms in the kernel, e.g. $x_2x_3'$, indicates that some coordinates of the feature vector must be mixed, i.e. contain different $x_i$'s.



## 4. The Kernel Perceptron Algorithm

The original Perceptron algorithm is given as follows
$$
\text{Perceptron }(\{(x^{(i)}, y^{(i)}), i = 1, ..., n\}, T):\\
\text{initialize }\theta = 0 \text{ (vector)};~~~~~~~~~~~~~~~~~~~~~\\
\text{for }t = 1, ..., T,~~~~~~~~~~~~~~~~~~~~~\\
\text{for }i=1,...,n,~~~~~~~~\\
~~~~~~\text{if }y^{(i)}(\theta \cdot x^{(i)} \leq 0),\\
~~~~~~~~~~~~~~~~~~~~~~\text{then update }\theta = \theta + y^{(i)}x^{(i)}
$$
For the kernel perceptron algorithm, $\theta$ is expressed as
$$
\theta = \sum _{j=1}^{n} \alpha _ j y^{(j)} \phi \left(x^{(j)}\right)
$$
where values of $\alpha _1, \ldots , \alpha _ n\,$ may vary at each step of the algorithm. In other words, we can reformulate the algorithm so we initialize and update $\alpha_j$s, instead of $\theta$.

The reformulated algorithm, or **kernel perceptron**，can be given in the following form
$$
\text{Perceptron }(\{(x^{(i)}, y^{(i)}), i = 1, ..., n\}, T):\\
\text{initialize }\alpha_1, \alpha_2, ...,\alpha_n \text{ to some values};\\
\text{for }t = 1, ..., T,~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\\
\text{for }i=1,...,n,~~~~~~~~~~~~~~~~~~~~~\\
~~~~~~~~~~~~~~~~~~~~~~~~~~~\text{if (Mistake Condition Expressed in } \alpha_j ) \\
~~\text{update }\alpha_j \text{ appropriately}
$$


### Initialization

Since $\theta = \sum _{j=1}^{n} \alpha _ j y^{(j)} \phi (x^{(j)})$, setting $\alpha_j = 0$ for all $j$ leads to $\theta = 0$.



### Update $\alpha_j$

Assuming there was a mistake in classifying the $i$th data point. i.e.
$$
y^{(i)}(\theta \cdot x^{(i)}) \leq 0
$$
In order to achieve the same update condition of the original algorithm as follows, 
$$
\theta = \theta + y^{(i)}\phi (x^{(i)})
$$
$\alpha _1, \alpha _2, ..., \alpha _ n$ should be updated following 
$$
\alpha_i = \alpha_i + 1
$$
Since by expanding $\theta$ in the last equation, it shows that only $\alpha_i$ gets updated
$$
\alpha _ i y^{(i)}\phi (x^{(i)}) + y^{(i)}\phi (x^{(i)}) = (\alpha _ i +1) y^{(i)}\phi (x^{(i)}).
$$


### The mistake condition

By plugging in $\theta = \sum _{j=1}^{n} \alpha _ j y^{(j)} \phi (x^{(j)})$, the equivalent conditions to $y^{(i)}(\theta \cdot \phi (x^{(i)})) \leq 0$ is as follows (Remember the kernel function $K$ is $K(x,x') = \phi (x) \phi (x').$)
$$
y^{(i)}\sum _{j=1}^{n} \alpha _ j y^{(j)} K(x^{j},x^{i}) \leq 0
$$


## 5. Kernel Composition Rules

1. $K(x, x')=1$ is a kernel function, as $\phi(x) = 1$.
2. Let $f: \R^d \rightarrow \R$ and $K(x, x')$ is a kernel. Then so is $\tilde{K}(x, x') = f(x) K(x, x')f(x')$, as $\tilde{\phi}(x) = f(x)\phi(x)$.
3. If $K_1(x,x')$ and $K_2(x,x')$ are kernels, then $K(x, x') = K_1(x,x') + K_2(x,x')$ is a kernel.
4. If $K_1(x,x')$ and $K_2(x,x')$ are kernels, then $K(x, x') = K_1(x,x') K_2(x,x')$ is a kernel.

> #### Exercise 21
>
> Let $x$ and $x′$ be two vectors of the same dimension. Use the the definition of kernels and the kernel composition rules from the video above to decide which of the following are kernels. (Note that you can use feature vectors $ϕ(x)$ that are not polynomial.)
> (Choose all those apply. )
>
> A. 1
>
> B. $x \cdot x'$
>
> C. $1 + x \cdot x'$
>
> D. $(1 + x \cdot x')^2$
>
> E. $exp(x + x'),$ for $x, x' \in \R$
>
> F. $min(x, x'),$ for $x,x' \in \Z$
>
> > **Answer**: ABCDE
>
> > **Solution**: 
> >
> > A. $\phi(x) = 1$
> >
> > B. $\phi(x) = x$
> >
> > C. Sum rule or $\phi(x) = [1,x]^T$ also works.
> >
> > D. Produce rule
> >
> > E. $\phi(x) = exp(x)$
> >
> > F. $\min$ can be a kernel given correct domain $\R+$.
> >
> > Let $\mathcal{X} \subset \mathbb{R}^d$ be a compact, finite dimensional vector space. A function $K : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is called a kernel if there exists an inner product space $V$, and a map $\varphi : \mathcal{X}  \to V$, such that $K(x, x') = \langle \varphi(x), \varphi(x') \rangle$ for all $x,x' \in \mathcal{X}$. Note this inner product doesn't need to have anything to do with the dot product in $\R^n$ for any $n$, it is happening in $V$.
> >
> > To see that $\min$ is a kernel, let us define $\mathcal{X} = \mathbb{R_{+}}$ and $V = L_2(\mathcal{X})$ the square-integrable functions on $\mathcal{X}$, and finally $\varphi(x) = \mathbb{1}_{[0, x]}$, the function which is 1 when its argument is in $[0,x]$, and 0 otherwise. $V$ is an inner product pace with the following inner product: $\langle f, g \rangle = \int_0^\infty f(t) g(t) dt$. Then, we have
> > $$
> > \begin{align}
> > K(x, y) &= \langle \mathbb{1}_{[0, x]}, \mathbb{1}_{[0, y]} \rangle \\ &= \int_0^\infty \mathbb{1}_{[0, x]}(t) \mathbb{1}_{[0, y]}(t) dt \\ &= \int_0^{\min(x, y)} 1 dt\\ &= \min(x, y)
> > \end{align}
> > $$
> > which makes $\min$ a kernel. In the question we are asked about $\mathcal{X} = \mathbb{Z}$ which is not $\R+$, so the argument does not apply.
> >

## 6. The Radial Basis Kernel

The **radial basis kernel** $K$ is defined as
$$
K(x,x') = e^{-\frac{1}{2} {||x-x'||}^2}
$$
The feature vectors can be **infinite dimensional**, this means that they have unlimited expressive power.

The decision boundary is defined by a set of $x$ that satisfies
$$
\{x: \sum^n_{j=1} \alpha_j y^{(j)} K(x^{(j)}, x)=0\}
$$
The decision boundary is non-linear in the original space while is linear in the infinite dimensional space.



# Additional Readings

A great lecture of kernel function: Lecture 15 - Kernel Methods

http://work.caltech.edu/telecourse.html

