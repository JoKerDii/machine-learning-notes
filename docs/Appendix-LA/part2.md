# Multiplying and Factoring Matrices 

#### Five equations:

* $A=LU$
  * By elimination, we got $L$ and $U$.
  * $L$: lower triangular matrix; $U$: upper triangular matrix
* $A = QR$
  * Gram-Schmidt's algorithm
  * $Q$: matrix with orthogonal (perpendicular) and orthonormal (unit vectors) columns
* $S = Q\Lambda Q^T$
  * $S$: symmetric; $Q$: orthogonal eigenvectors; $\Lambda$: diagonal eigenvalue matrix
* $A = X\Lambda X^{-1}$
* $A = U\sum V^T$
  * Singular Value Decomposition (SVD)
  * $U$: orthogonal matrix; $V$: orthogonal matrix; $\sum$: diagonal matrix

> #### Example 1
>
> Use $S = Q\Lambda Q^T$ to explain **Matrix Multiplication**.
>
> $(Q\Lambda)(Q^T)$ = sum of rank 1 = $\lambda_1 q_1 q_1^T + \lambda_2 q_2 q_2^T + ... + \lambda_n q_n q_n^T$, [**spectral theorem**]
>
> Let $S = \lambda_1 q_1 q_1^T + \lambda_2 q_2 q_2^T + ... + \lambda_n q_n q_n^T$, as vectors of $Q$ are orthogonal, we got
> $$
> S q_1 = \lambda_1 q_1 q_1^T q_1 + 0 + ... + 0
> $$
> As $q_1^T q_1 = \|q_1\|^2 = 1$, we got
> $$
> S q_1 = \lambda_1 q_1\\
> $$
> The matrix $S$ is correct, since it got the right eigenvectors $q$ and right eigenvalues $\lambda$.

-----

#### Four fundamental subspaces: ($A$ matrix ($m \times n$), rank $r$ )

* Column space $C(A)$ : dimension = $r$
* Row space $C(A^T)$: dimension = $r$
* Null space $N(A)$: dimension = $n-r$
* Null space $N(A^T)$: dimension = $m-r$

Recall that **null space** is a set of solution to $Ax = 0$.

Key points: 

* $C(A^T)$ and $N(A)$ are in $R^n$ space, and they are orthogonal.
* $C(A)$ and $N(A^T)$ are in $R^m$ space, and they are orthogonal.

> #### Example 2
>
> Given $A = \begin{bmatrix}  1~~~2~~~3\\ 4~~~7~~~9 \end{bmatrix}$ is a ($2\times 3$) matrix.
>
> $C(A^T)$ is the row space, each row has 3 elements or in 3 dimension. $N(A) = \begin{bmatrix}  x_1\\ x_2 \\x_3 \end{bmatrix}$ that satisfies $\begin{bmatrix}  1~~~2~~~3\\ 4~~~7~~~9 \end{bmatrix} \begin{bmatrix}  x_1\\ x_2 \\ x_3 \end{bmatrix} = 0$, where every row of $A$ is orthogonal to vector $x$. $C(A^T)$ and $N(A)$ are in $R^3$.
>
> $C(A)$is the column space, each column has 2 elements or in 2 dimension. $N(A^T) = \begin{bmatrix}  x_1\\ x_2 \end{bmatrix}$ that satisfies $\begin{bmatrix}  1~~~4\\ 2~~~7\\ 3~~~9 \end{bmatrix}\begin{bmatrix}  x_1\\ x_2 \end{bmatrix} = 0$, where every row of $A^T$ is orthogonal to vector $x$. $C(A)$ and $N(A^T)$ are in $R^2$.



