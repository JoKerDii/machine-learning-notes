# Nonlinear Classification

There are topics and exercises



## 1. Higher Order Feature Vectors

We can use linear classifiers to make non-linear predictions. The easiest way to do this is to first map all the examples $x∈\R^d$ to different feature vectors $ϕ(x)∈\R^p$ where typically $p$ is much larger than $d$. We would then simply use a linear classifier on the new (higher dimensional) feature vectors, pretending that they were the original input vectors. As a result, all the linear classifiers we have learned remain applicable, yet produce non-linear classifiers in the original coordinates.

> #### Exercise 18
>
> Given the training examples with $x=[x^{(t)}_1,x^{(t)}_2]∈\R^2$above, where a boundary between the positively-labeled examples and the negatively-labeled examples is an ellipse , which of the following feature vector(s) $ϕ(x)$ will guarantee that the training set $\{(ϕ(x(t)),y(t)),t=1,…,n\}$ (where $y(t)$ are the labels) are linearly separable?
> (Choose all that apply.)
>
> ![lec6_exercise](../assets/images/U2/lec6.svg)
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
> > **Answer**: E. $\phi (x)=[x_1,\, x_2,\, x_1x_2,\, x_1^2,\, x_2^2]^ T$
>
> > **Solution**: 
> >
> > Since a possible boundary is an ellipse, and we recall from geometry that the equation of any ellipse can be given as $\theta _1 x_1+\theta _2 x_2+\theta _3x_1x_2+\theta _4x_1^2+\theta _5 x_2^2+ \theta _0\, =\, \theta \cdot [x_1,\, x_2,\, x_1x_2,\, x_1^2,\, x_2^2]^ T\, +\theta _0=\, 0,$
> >
> > for some $\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5]$, we define the feature map to be $\phi (x)=[x_1,x_2,x_1x_2,x_1^2,x_2^2]^ T$. Thus the linear decision boundary in $\phi$-coordinates:
> > $$
> > \theta \cdot \phi (x)+\theta _0 = 0 \Longleftrightarrow \theta \cdot [x_1,x_2,x_1x_2,x_1^2,x_2^2]^ T+\theta _0 = 0
> > $$
> > A. $\phi(x)$ maps $x$ to $x$ itself in $\R^2$
> >
> > C/D. $\phi (x)=[x_1,x_2,x_1^2+x_2^2]^ T$ gives decision boundaries of the form: 
> > $$
> > \theta \cdot [x_1,x_2,x_1^2+x_2^2]^ T+\theta _0\, = \theta _1 x_1+\theta _2 x_2+\theta _3\left(x_1^2+x_2^2\right)+ \theta _0\, =\, 0
> > $$
> > which are circles in $(x_1, x_2)$ plane.



## 2. Non-linear Classification

By mapping examples into a feature representation and performing linear classification in the new feature coordinates, we get a **nonlinear classifier**. We can add more **linearly independent** feature coordinates and get more powerful feature representations and more powerful classifiers.

* **Example**: we have vector $x = [x_1, x_2]$ which is 2 dimensional and lives in $\R^2$. If we add the second order of features, we will get $[x_1,x_2,x_1^2,x_2^2,x_1x_2]$ which is 5 dimensional.

> Exercise 19
>
> 

