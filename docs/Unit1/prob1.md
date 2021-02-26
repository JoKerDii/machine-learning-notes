# Problem 1

There are 5 questions.

## Novikoff Theorem [[paper](https://arxiv.org/pdf/1305.0208.pdf)]

Assume:

- There exists $\theta^*$ such that $\frac{y^{(i)} (\theta^* x^{(i)})}{\|\theta^*\|} \geq \gamma$ for all $i = 1, ..., n$ and some $\gamma>0$. [which implies that the data is linearly separable.]
- All the examples are bounded $\|x^{(i)}\| \leq R, i=1,⋯,n$ .

Then the number $k$ of updates made by the perceptron algorithm is bounded by $\frac{R^2}{\gamma^2}$.

> #### Question 1: 
>
> Based on this theorem, what are the factors that constitute the bound on the number of mistakes made by the algorithm?
>
> A. Iteration order
>
> B. Maximum margin between positive and negative data points
>
> C. Maximum norm of data points
>
> D. Average norm of data points
>
> > **Answer**: BC
>
> > **Solution**: 
> >
> > A. iteration order would affect relative convergence speed but does not constitute the bounds in the maximum number of mistakes. 
> >
> > C. the maximum norm of the data points is the length of the point-vector that is furthest from the origin. So it would be the longest point -  $[x_1, ..., x_i]$.
> >
> > D. we can always scale an easy dataset to achieve the same average norm of data points with the same number of mistakes.



> #### Question 2: 
>
> If we want to establish an adversarial procedure to maximize the number of mistakes the perceptron algorithm makes, what are possible solutions?
>
> A. Exhaustive search the worst ordering of iterating data points for updating parameters
>
> B. Dynamic programming the worst ordering of iterating data points for updating parameters
>
> C. Greedily select the data point with the maximum norm
>
> > **Answer**: AB
>
> > **Solution**: 
> >
> > A. Since the algorithm can converge in finite adjustments, there are only finitely possible iteration orders, thus A is correct, exhaustive search can always find the worst ordering. 
> >
> > B. Given any prior mistakes made by the algorithm, the maximum number of mistakes should be prior mistakes plus maximum future mistakes. Hence, optimal substructure exist and dynamic programming can be applied.
> >
> > C. Choosing the data point with maximum norm does not maximize the number of mistakes. While the data point with the maximum norm gives an upper bound on the number of updates, the data point itself is not necessarily a bad point to start with. Since we are looking at the good vs bad ordering here, the other points also have effects.

The theorem above provides an upper bound on the number of steps of the Perceptron algorithm and implies that it indeed converges. In fact, given a set of training examples that are linearly separable through the origin, we show that the initialization of $\theta$ does not impact the perceptron algorithm's ability to eventually converge.

If $\theta$ is initialized to 0, by induction:
$$
\theta ^{(k+1)} \cdot \frac{\theta ^* }{ \| \theta ^*\| } =(\theta ^{(k)} +\  y^{(i)}x^{(i)} )\cdot \frac{\theta ^* }{ \| \theta ^*\| } \geq (k+1)\gamma
$$
we can show that:
$$
\theta^{(k)} \frac{\theta^*}{\|\theta^*\|} \geq k \gamma
$$
If we initialize $\theta$ to a general (not necessarily 0) $\theta^{(0)}$, then:
$$
\theta ^{(k)} \cdot \frac{\theta ^*}{ \| \theta ^*\| } \geq a + k \gamma
$$

> #### Question 3:
>
> Determine the formulation of $a$ in terms of $\theta^*$ and $\theta^{(0)}$.
>
> > **Answer**: $a = \theta^{0} \cdot \frac{\theta^{*}}{\| \theta^{*}\|}$
>
> > **Solution**: Because  $\theta^{(k)} \frac{\theta^*}{\|\theta^*\|} \geq \theta^{(k-1)} \frac{\theta^*}{\|\theta^*\|} + \gamma$



If we initialize $\theta^{(0)}$ to 0, then by induction:
$$
\| \theta ^{(k+1)}\| ^2 \leq \| \theta ^{(k)} + y^{(i)}x^{(i)} \| ^2 \leq \| \theta ^{(k)} \| ^2 + R^2
$$
we got:
$$
\| \theta ^{(k)}\| ^2 \leq k R^2
$$
If we initialize $\theta$ to a general $\theta^{(0)}$, then:
$$
\| \theta ^{(k)}\| ^2 \leq kR^2 + c^2
$$

> #### Question 4:
>
> Determine the formulation of $c^2$ in terms of $\theta^{(0)}$.
>
> > **Answer**: $c^2 = \| \theta^{0} \|^2$ 
>
> > **Solution**:  Because $\| \theta ^{(k)}\| ^2  \leq \| \theta ^{(k-1)}\| ^2 + R^2 $, $\| \theta ^{(k)}\| ^2  \leq \| \theta ^{(0)}\| ^2 + k R^2 $



By applying $\sqrt{x^2 + y^2} \leq \sqrt{(x + y)^2}$ if $x, y$ > 0,

We can derive the inequality
$$
\| \theta ^{(k)}\| \leq c + \sqrt{k}R
$$
If $\theta$ is initialized to 0, we then use the fact that $\displaystyle 1 \geq \frac{\theta ^{(k)}}{\| \theta ^{(k)}\| } \cdot \frac{\theta ^*}{\| \theta ^*\| }$ to get the upper bound $\displaystyle k\le \frac{R^2}{\gamma ^2}$.

> #### Question 5:
>
> In the case where we initialize $\theta$ to a general $\theta^{(0)}$, use the inequality for $\theta ^{(k)} \cdot \frac{\theta ^*}{ \| \theta ^*\| }$ above and the inequality $\| \theta ^{(k)}\| \leq c + \sqrt{k}R$ to derive a bound on the number of iteration $k$. (Use the larger root of a quadratic equation to obtain the upper bound.)
>
> > **Answer**: $k \leq \frac{(R + \sqrt{(R^2 - 4 \cdot \gamma \cdot (a-c))})^2}{(4 \cdot \gamma^2)}$
>
> > **Solution**: Solving:
> > $$
> > 1 \geq \theta ^ k \cdot \frac{\theta ^* }{\| \theta ^ k\| \| \theta ^*\| } \geq \frac{a+k\gamma }{c+\sqrt{k}R} \\
> > \Longrightarrow a+k\gamma -c\leq \sqrt{k} R \\
> > $$



**Note**: 

The convergence of he perceptron algorithm does not depend on the initialization, in other words, the end performance on the training set must be the same. However, this does not mean that the resulting $\theta$s are the same regardless of the initialization. In fact, any distinct $\theta$ that can separate the data are valid solutions, so there are infinitely many different valid correct $\theta$ in general given that the data can be separated by more than 1 line.

This also implies that the performance on a test set would be different, as two different $\theta$ would always make different predictions for some test points.

