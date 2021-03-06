# Recommender System

There are 4 topics and 1 question.

## 1. K-Nearest Neighbor Method

Let $m$ be the number of movies and $n$ the number of users. The ranking $Y_{ai}$ of a movie $i∈\{1,...,m\}$ by a user $a∈\{1,...,n\}$ may already exist or not. The **sparse** movie matrix is represented as follows. Our goal is to predict $Y_{ai}$ in the case when $Y_{ai}$ does not exist.
$$
Y_{n,m} =
 \begin{pmatrix}
  y_{1,1} & y_{1,2} & \cdots & y_{1,m} \\
  y_{2,1} & y_{2,2} & \cdots & y_{2,m} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  y_{n,1} & y_{n,2} & \cdots & y_{n,m} 
 \end{pmatrix}
$$
The $K$-Nearest Neighbor method makes use of ratings by $K$ other similar users when predicting $Y_{ai}$. Let $KNN(a)$ be the set of $K$ users "similar" to user $a$, and let $sim(a,b)$ be a **similarity measure** between users $a$ and $b \in KNN(a)$. The $K$-Nearest Neighbor method predicts a ranking $Y_{ai}$ to be

$$
\widehat{Y}_{ai} = \displaystyle \frac{\displaystyle \sum _{b \in \text {KNN}(a)} \text {sim}(a,b) Y_{bi}}{\displaystyle \sum _{b \in \text {KNN}(a)} \text {sim}(a,b)}.
$$

The similarity measure $sim(a,b)$ could be any **distance function** between the feature vectors $x_a$ and $x_b$ of users $a$ and $b$, e.g. the **Euclidean Distance** $\|x_a - x_b\|$ and the cosine similarity $cos \theta = \frac{x_a \cdot x_b}{\|x_a\| \|x_b\|}$. 

A drawback of this method is that the success of the $K$-Nearest Neighbor method depends on the choice of the similarity measure. **Collaborative filtering** will free us from the need to define a good similarity measure.

## 2. Collaborative Filtering: the Naive Approach

Let $Y$ be a matrix with $n$ row and $m$ columns whose $(a,i)$th entry $Y_{ai}$ is the rating by user $a$ of movie $i$ if this rating has already been given, and blank if not. Our goal is to come up with a matrix $X$ that has no blank entries and whose $(a,i)$th entry $X_{ai}$ is the prediction of the rating user a will give to movie $i$.

Let $D$ be the set of all $(a,i)$'s for which a user rating $Y_{ai}$ exists, i.e. $(a,i)∈D$ if and only if the rating of user a to movie $i$ exists. A naive approach to solve this problem would be to minimize the following objective
$$
J(X) = \sum _{a,i \in D} \frac{(Y_{ai} - X_{ai})^2}{2} + \frac{\lambda }{2} \sum _{(a,i)} X_{ai}^2
$$
which consists of the sum of the squared errors and a regularization term.

Compute the derivative $\frac{ \partial J}{\partial X_{ai}}$ of the objective function $J(X)$
$$
\frac{ \partial J}{\partial X_{ai}} = \frac{\partial ((\frac{Y_{ai}-X_{ai}}{2})^2 - \frac{\lambda}{2}X_{ai}^2)}{\partial X_{ai}} = 0\\
$$
For (any fixed) $(a,i) \in D$,
$$
{∂J\over∂X_{ai}}= X_{ai}-Y_{ai}+\lambda \cdot X_{ai} = 0\\
X_{ai} = \frac{Y_{ai}}{1+\lambda}
$$
For (any fixed) $(a,i) \notin D$,
$$
{∂J\over∂X_{ai}}= \lambda \cdot X_{ai}\\
X_{ai} = 0
$$

## 3. Collaborative Filtering with Matrix Factorization

In addition to minimizing the objective and obtain the trivial solution of $X$, in the **collaborative filtering** approach, we impose an additional constraint on $X$.
$$
X = UV^T
$$
for some $n \times d$ matrix $U$ and $d \times m$ matrix $V^T$. The advantage is that the number of parameters are reduced from $O(n \times m)$ to $O(n \times d + d \times m)$. 

The number $d$ is the **rank** of the matrix $X$. The rank of a matrix corresponds to ways to factorize the matrix.

## 4. Alternating Minimization

Now $X_{ai} = u_a v_i$, we are going to find $U$ and $V$ that minimize our new objective
$$
\displaystyle J(u,v) = \sum _{(a,i) \in D} \frac{(Y_{ai} - \big [UV^ T\big ]_{ai})^2}{2} + \frac{\lambda }{2} \left(\sum _{a,k} U_{ak}^2 + \sum _{i,k} V_{ik}^2\right).
$$
To simplify the problem, we fix $U$ and solve for $V$, then fix $V$ to be the result from the previous step and solve for $U$, and repeat this alternate process until we find the solution.

> #### Exercise 22: 
>
> Let $V$ be fixed first as $V = \begin{bmatrix} 2\\ 7\\ 8 \end{bmatrix}$, $Y = \begin{bmatrix} 5~~~?~~~7\\ 1~~~2~~~? \end{bmatrix}$. $U = \begin{bmatrix} u_1\\ u_2 \end{bmatrix}$. What is the solution for $U$?
>
> > **Answer**: $U = \begin{bmatrix} \frac{66}{\lambda + 68}\\ \frac{16}{\lambda + 53} \end{bmatrix}$
>
> > **Solution**: 
> > $$
> > UV^T = \begin{bmatrix} u_1\\ u_2 \end{bmatrix} \begin{bmatrix} 2~~~ 7~~~ 8 \end{bmatrix} = \begin{bmatrix} 2u_1~~~7u_1~~~8u_1\\ 2u_2~~~7u_2~~~8u_2 \end{bmatrix}\\
> > \frac{\partial}{\partial u_1} [\frac{(5-2u_1)^2}{2} + \frac{(7-8u_1)^2}{2} + \frac{\lambda}{2} u_1^2] = -66 + (68 + \lambda) u_1 = 0\\
> > \frac{\partial}{\partial u_2} [\frac{(1-2u_1)^2}{2} + \frac{(2-7u_2)^2}{2} + \frac{\lambda}{2} u_2^2] = -16 + (53 + \lambda) u_2 = 0
> > $$
> >
> > $$
> > u_1 = \frac{66}{\lambda + 68}; u_2 = \frac{16}{\lambda + 53}
> > $$

