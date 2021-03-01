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









