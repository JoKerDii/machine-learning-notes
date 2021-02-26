# Problem 4

There are 2 questions.

## Perceptron Updates

In this problem, we aim to understand the convergence of perceptron algorithm and its relation to the ordering of the training sample.

Consider a set of $n=d$ labeled $d$−dimensional feature vectors,$ \{(x(t),y(t)),t=1,…,d\} (d>0)$ defined as follows:
$$
\begin{aligned}
x^{(t)}_ i & = cos(\pi t) \text{ if } i = t\\
x^{(t)}_ i & = 0 \text{ otherwise } \\
\end{aligned}
$$
Assume that we initialize $\theta = 0$ , $\theta \cdot x = 0$ is treated as a mistake, and there is no offset.

> #### Question 12:
>
> Consider the case with $d=3$. Also assume that all the feature vectors are positively labelled. Let $P$ denote the plane through the three points in a 3-d space whose vector representations are given by the feature vectors $x^{(1)},x^{(2)},x^{(3)}$.
>
> Let $\hat{\theta}$ denote the value of $θ$ after perceptron algorithm converges for this example. Let $v$ denote the vector connecting the origin and $\hat{\theta}$. Which of the following options is true regarding the vector represented by $\hat{\theta}$.
>
> A. $v$ is parallel to the plane $P$
>
> B. $v$ is perpendicular to the plane $P$ and pointing away from it
>
> C. $v$ is perpendicular to the plane $P$ and pointing towards it
>
> D. $\hat{\theta}$ lies on the plane $P$.
>
> > **Answer**: C
>
> > **Solution**:
> >
> > From the problem setting we get points: $x^{(1)} = [-1,0,0],x^{(2)} = [0,1,0],x^{(3)} = [0,0,-1]$.
> >
> > Update the $\theta$ until the algorithm converges we got $\theta =  \begin{bmatrix} -1 \\ 1\\ -1\end{bmatrix}$.
> >
> > Compute the equation of the plane, we assume any point on the plane $v = [x,y,z]$:
> >
> > $u_1 = x^{(2)} x^{(1)}  = x^{(1)} - x^{(2)} = [-1,-1,0]  $
> >
> > $u_2 = x^{(2)} x^{(3)}  = x^{(3)} - x^{(2)} = [0,-1,-1]$
> >
> > $u_3 = x^{(2)} v  = v - x^{(2)} = [x,y-1,z]$
> >
> > Thus the plane can be represented by
> > $$
> > \begin{vmatrix}  x &  -1 &  0 \\ y-1 &  -1 &  -1 \\ z &  0 &  -1 \end{vmatrix} = 0
> > $$
> > Solve this equation by
> > $$
> > x \begin{vmatrix}   -1 &  -1 \\  0 &  -1 \ \end{vmatrix} - (y-1) \begin{vmatrix}   -1 &  0 \\  0 &  -1 \ \end{vmatrix} + z \begin{vmatrix}   -1 &  0 \\  0 &  -1 \ \end{vmatrix} = 0
> > $$
> > we get
> > $$
> > x - y + z = -1
> > $$
> > where a norm of the plane is $n = \begin{bmatrix} 1 \\ -1\\ 1\end{bmatrix}$ .
> >
> > Equivalently,
> > $$
> > \begin{pmatrix}  x \\ y \\ z \end{pmatrix} \cdot \begin{pmatrix}  -1 \\ 1 \\ -1 \end{pmatrix} = 1
> > $$
> > Therefore, $v$ is perpendicular to $P$ and pointing towards it.



> #### Question 13:
>
> Please choose the correct answer:
>
> A. Perceptron algorithm will make **exactly** $d$ updates to $θ$ **regardless of** the order and labels of the feature vectors.
>
> B. Perceptron algorithm will make **at least** $d$ updates to $θ$ with the exact number of updates **depending on** the ordering of the feature vectors presented to it.
>
> C. Perceptron algorithm will make **at least** $d$ updates to $θ$ with the exact number of updates **depending on** the ordering of the feature vectors presented to it and their labeling.
>
> D. Perceptron algorithm will make **at most** $d$ updates to $θ$ with the exact number of updates **depending on** the ordering of the feature vectors presented to it and their labeling.
>
> > **Answer**: A
>
> > **Solution**: 
> >
> > After the $i$th update, we add $y^{(i)}x^{(i)}$ to $θ^{(i−1)}$. After $d$ updates,
> > $$
> > \theta ^{(d)} = \sum _{i=1}^ d y^{(i)}x^{(i)}
> > $$
> > We check whether $y^{(t)}θ^{(d)}⋅x^{(t)}>0$ for all $t$ to ensure there are no mistakes. Notice that the only non-zero term of the dot product occurs when $i=t$. Thus,
> > $$
> > y^{(t)}\theta ^{(d)} \cdot x^{(t)} = (y^{(t)})^2 \left\|  x^{(t)} \right\| ^2 > 0
> > $$
> > for all $t=1,2,⋯,d$. After $d$ updates, all points are classified correctly. Therefore, the perceptron algorithm will make exactly $d$ updates regardless of the order and the labels of the feature vectors.

