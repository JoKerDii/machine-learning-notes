# Multiplying and Factoring Matrices 

**Inner product** (row times columns) produce each of the numbers in $AB = C$. **Outer product** is a way to multiply $AB$ by columns of $A$ timing rows of $B$.

An example: 
$$
uv^T = \begin{bmatrix}  2\\ 2 \\ 1 \end{bmatrix} \begin{bmatrix}  3 & 4 & 6 \end{bmatrix} = \begin{bmatrix}  6&8&12\\ 6&8&12 \\ 3&4&6 \end{bmatrix}
$$
Some conclusions:

* $uv^T$ is a rank one matrix
* All columns of $uv^T$ are multiples of $u = \begin{bmatrix}  2\\ 2 \\ 1 \end{bmatrix}$. All rows are multiples of $v^T = \begin{bmatrix}  3 & 4 & 6 \end{bmatrix}$.
* The column space of $uv^T$ is 1D: the line in the direction of $u$. The row space of it is 1D: the line through $v$.
* The row space of $uv^T$ is the column space of $A^T$, i.e. $C(A^T)$.
* Row rank = Column rank; r independent columns = r independent rows

Importance of outer product: 

* By outer product, we are looking for the important part of a matrix $A$.

-----

#### Five importance factorizations:

* $A=LU$
  * From **elimination**.
  * $L$: lower triangular matrix; $U$: upper triangular matrix
* $A = QR$
  * From **orthogonalizing**
  * Gram-Schmidt's algorithm
  * $Q$: matrix with **orthogonal** (perpendicular, $Q^T = Q^{-1}$) and **orthonormal** (unit vector, $Q^TQ = I$) columns ; $R$ is the upper triangular
* $S = Q\Lambda Q^T$
  * From eigenvalues $\lambda_1, ..., \lambda_n$ of a symmetric matrix $S = S^T$.
  * $S$: symmetric matrix; $Q$: orthogonal eigenvectors; $\Lambda$: diagonal eigenvalue matrix
* $A = X\Lambda X^{-1}$
  * From **diagonalization** when $A$ is $n$ by $n$ with $n$ independent eigenvectors.
  * **Eigenvalues** of $A$ on the diagonal of $\Lambda$. **Eigenvectors** of $A$ in the columns of $X$.
* $A = U\sum V^T$
  * From Singular Value Decomposition (SVD) of any matrix $A$.
  * $U$: orthogonal matrix; $V$: orthogonal matrix; $\sum$: diagonal matrix with singular values $\sigma_1, ..., \sigma_r$.

> #### Example 3
>
> Use $S = Q\Lambda Q^T$ to explain **Matrix Multiplication**.
>
> $(Q\Lambda)(Q^T)$ = sum of rank 1 = $\lambda_1 q_1 q_1^T + \lambda_2 q_2 q_2^T + ... + \lambda_n q_n q_n^T$, [**spectral theorem**]
>
> Let $S = \lambda_1 q_1 q_1^T + \lambda_2 q_2 q_2^T + ... + \lambda_n q_n q_n^T$, which is known as "**rank one pieces**", since vectors of $Q$ are orthogonal, we got
> $$
> S q_1 = \lambda_1 q_1 q_1^T q_1 + 0 + ... + 0
> $$
> As $q_1^T q_1 = \|q_1\|^2 = 1$, we got
> $$
> S q_1 = \lambda_1 q_1\\
> $$
> The matrix $S$ is correct, since it got the right eigenvectors $q$ and right eigenvalues $\lambda$.

Some conclusions:

* Symmetric matrix $S$: $S^T = S$, all $s_{ij} = _{ji}$. Orthogonal matrix $Q$: $Q^T = Q^{-1}$, all $q_i \cdot q_j = \begin{cases} 0, & \text{for }\ i \neq j \\ 1, & \text{for }\ i = j \end{cases}$. Diagonal matrix $\Lambda$ contains real eigenvalues $\lambda_1$ to $\lambda_n$. Every real symmetric matrix $S$ has $n$ **orthonormal eigenvectors** $q_1$ to $q_n$, and $n$ real eigenvalues.

* When multiplied by $S$, the eigenvectors keep the same direction. They are just rescaled by the number $\lambda$.

* Eigenvector $q$ and eigenvalue $\lambda$: $Sq = \lambda q$, from which we got $SQ = Q \Lambda$.
  $$
  SQ = S \begin{bmatrix} q_1 & ... & q_n \end{bmatrix} = \begin{bmatrix}  \lambda_1 q_1 & ... & \lambda_n q_n \end{bmatrix} = \begin{bmatrix} q_1 & ... & q_n \end{bmatrix} \begin{bmatrix}  \lambda_1 & & \\  &...&  \\  & & \lambda_n \end{bmatrix} = Q \Lambda
  $$
  Multiply $SQ = Q \Lambda$ by $Q^{-1} = Q^T$ to get $S = Q \Lambda Q^T$ [**spectral theorem**], which is a symmetric matrix. 

* Each eigenvalue $\lambda_k$ and each eigenvector $q_k$ contribute a "rank one piece" $\lambda_k q_k q_k^T$ to $S$.
  $$
  S = (Q \Lambda)Q^T = (\lambda_1 q_1) q_1^T + (\lambda_2 q_2) q_2^T + ... + (\lambda_n q_n) q_n^T
  $$
  The columns of $Q \Lambda$ are $\lambda_1 q_1$ to $\lambda_n q_n$, implying that when multiplying a matrix on the right by the diagonal matrix $\Lambda$, you multiply its columns by the $\lambda$'s.

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

> #### Example 4
>
> Given $A = \begin{bmatrix}  1~~~2~~~3\\ 4~~~7~~~9 \end{bmatrix}$ is a ($2\times 3$) matrix.
>
> $C(A^T)$ is the row space, each row has 3 elements or in 3 dimension. $N(A) = \begin{bmatrix}  x_1\\ x_2 \\x_3 \end{bmatrix}$ that satisfies $\begin{bmatrix}  1~~~2~~~3\\ 4~~~7~~~9 \end{bmatrix} \begin{bmatrix}  x_1\\ x_2 \\ x_3 \end{bmatrix} = 0$, where every row of $A$ is orthogonal to vector $x$. $C(A^T)$ and $N(A)$ are in $R^3$.
>
> $C(A)$is the column space, each column has 2 elements or in 2 dimension. $N(A^T) = \begin{bmatrix}  x_1\\ x_2 \end{bmatrix}$ that satisfies $\begin{bmatrix}  1~~~4\\ 2~~~7\\ 3~~~9 \end{bmatrix}\begin{bmatrix}  x_1\\ x_2 \end{bmatrix} = 0$, where every row of $A^T$ is orthogonal to vector $x$. $C(A)$ and $N(A^T)$ are in $R^2$.



