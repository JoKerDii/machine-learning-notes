# The Column Space of A Contains All Vectors Ax

$Ax$ is a linear combination of the columns of $A$.  

$\implies$ The **column space** of $A$ contains all vectors $Ax$. i.e. $Ax$ is the column space. 

Formally, **The combinations of the columns fill out the column space of $A$.** 

(Note that $ABCx$ is also the column space of $X$, as $ABCx = A(BCx) = Ax$).

> #### Example 1: 
>
> All combinations $Ax = x_1a_1+x_2a_2$ produce what part of the full 3D space？ Those vectors produces a plane.
>
> * The plane contains the complete lines in the direction of $a_1$, since every vector $x_1a_1$ is included.
> * The plane contains the line of all vectors $x_2a_2$ in the direction of $a_2$.
> * The plane contains the sum or any vector on one line plus any vector on the other line. This addition fills out an infinite plane containing the two lines. But it does not fill out the whole 3D space $R^3$.
> * **The combinations of the columns fill out the column space of $A$**. Here the column space is a plane.

**$b = (b1,b2,b3)$ is in the column space of $A$ exactly when $Ax = b$ has a solution $(x_1, x_2)$.**

The solution $x$ shows how to express the right side $b$ as a combination $x_1a_1 + x_2a_2$ of the columns.

-----

Here is a total list of all possible column spaces inside $R^3$, dimension 0,1,2,3, with $a_1, a_2, a_3$ to be independent: 

* The zero vector $(0,0,0)$
* A line of all vectors $x_1a_1$
* A plane of all vectors $x_1a_1 + x_2a_2$
* The whole $R^3$ with all vectors $x_1a_1 + x_2a_2 + x_3a_3$

In linear algebra:

* Three independent columns in $R^3$ produce an **invertible matrix**: $AA^{-1} = A^{-1}A = I$.
* $Ax=0$ requires $x = (0,0,0)$. Then $Ax=b$ has exactly one solution $x = A^{-1}b$.

------

By factoring $A$ into $C$ times $R$, we can get:

* The basis for the column space of $A$ or $C$, and the basis for the row space of $R$. Formally, **A basis for a subspace is a full set of independent vectors: All vectors in the space are combinations of the basis vectors.**
* The rank of the matrix $A$ or $C$ (the same), which counts the independent columns.. **The rank of a matrix $A$ is the dimension of the column space of A or C (same space)**. 
* The dimension of the subspace
> ####  Example 2:
>
> $$
> A = \begin{bmatrix}  2&1&3\\ 3&1&4 \\ 5&7&12 \end{bmatrix} = \begin{bmatrix}  2&1\\ 3&1 \\ 5&7 \end{bmatrix} \begin{bmatrix}  1&0&1\\ 0&1&1 \end{bmatrix} = CR
> $$
>
> $C = \begin{bmatrix}  2~~~1\\ 3~~~1 \\ 5~~~7 \end{bmatrix}$ is the column space. Each column of it is the basis for the column space of $A$.  Each row of $\begin{bmatrix}  1~~~0~~~1\\ 0~~~1~~~1 \end{bmatrix}$ is the basis for the row space. Two columns in $C$ corresponds to two rows in $R$. This proves that row rank equals column rank. 

The key idea is $A = CR$,  which is the first idea of **factorization** of a matrix. Combinations of the columns of $C$ produce the columns of $A$, then $A = CR$ stores this information as a matrix multiplication.  Some properties:

* The shape is $(m \times n) = (m \times r)(r \times n)$.
* The columns of $C$ should be **independent**.  **The number of independent columns equals the number of independent rows**. Thus **row rank equals column rank**.
* The $R$ matrix is $R = rref(A)$ which is named "*row-reduced echelon form of $A$ (without zero rows)*".
* The big factorization for data science is the "**SVD**" of $A$ when the first factor $C$ has $r$ **orthogonal** columns and the second factor $R$ has $r$ **orthogonal** rows. When $C$ takes columns directly from $A$, and $R$ takes rows directly from $A$, those matrices preserve properties that are lost in the more famous $QR$ and $SVD$ factorizations. Where $A = QR$ and $A = U\sum V^T$ involve orthogonalizing the vectors, $C$ and $R$ keep the original data:
  * If A is **nonnegative**, so are $C$ and $R$.
  * If $A$ is **sparse**, so are $C$ and $R$.

------

The true equation is actually $A = CMR$. $M$ is $r$ by $r$ invertible "mixing matrix".

Elimination reduces matrix $A$ to $R$ as follows. 
$$
A = \begin{bmatrix}  1~~~3~~~8\\ 1~~~2~~~6 \\ 0~~~1~~~2 \end{bmatrix}\rightarrow ~\rightarrow R = \begin{bmatrix}  1~~~0~~~2\\0~~~1~~~2\\ 0~~~0~~~0 \end{bmatrix} = rref(A)
$$
Those rows of $R$ do not come directly from $A$, $R$ actually has the from $MR$. An example is 
$$
A = \begin{bmatrix}  2~~~4\\ 3~~~6 \end{bmatrix} = \begin{bmatrix}  2\\ 3  \end{bmatrix} \begin{bmatrix}  1~~~2 \end{bmatrix} =\begin{bmatrix}  2\\ 3  \end{bmatrix} [\frac{1}{2}]\begin{bmatrix}  2~~~4 \end{bmatrix}= CMR
$$
$C$ and $R$ are not necessarily squared. They have one-sided inverses. We invert $C^TC$ and $RR^T$. To find $M$:
$$
\begin{aligned}
A & = CMR\\
C^TAR^T & = C^TCMRR^T\\
M & = (C^TC)^{-1}(C^TAR^T)(RR^T)^{-1}
\end{aligned}
$$
*Recall that the computation of **inverse**: (an example of 2 by 2 matrix)
$$
\begin{bmatrix}  a~~~b\\ c~~~d \end{bmatrix}^{-1} = \frac{1}{ad-bc}\begin{bmatrix}  d&-b\\ -c&a \end{bmatrix}
$$

