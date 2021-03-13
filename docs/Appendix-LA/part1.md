# Part 1 - Column Space

The **column space** of $A$ contains all vectors $Ax$, i.e. $Ax$ is the column space. Note that $ABCx$ is also the column space of $X$, as $ABCx = A(BCx) = Ax$.

The **rank** is the dimension of the column space. 

$A = \begin{bmatrix}  1~~~3~~~8\\ 1~~~3~~~8 \\ 1~~~3~~~8 \end{bmatrix}$ has $C(A)$ = Line; $rank(A) = 1$

The **row space** is all combination of rows. $C(A^T) = $ row space of $A$.

------

$A = \begin{bmatrix}  2~~~1~~~3\\ 3~~~1~~~4 \\ 5~~~7~~~12 \end{bmatrix} = \begin{bmatrix}  2~~~1\\ 3~~~1 \\ 5~~~7 \end{bmatrix} \begin{bmatrix}  1~~~0~~~1\\ 0~~~1~~~1 \end{bmatrix}$

where $\begin{bmatrix}  2~~~1\\ 3~~~1 \\ 5~~~7 \end{bmatrix}$ is the column space, each column of which is the basis for the column space of $A$. Each row of $\begin{bmatrix}  1~~~0~~~1\\ 0~~~1~~~1 \end{bmatrix}$ is the basis for the row space. It is called the  "row reduced echelon form of matrix". This proves that row rank equals to column rank.

The key idea is $A = CR$ or $A = CUR$, the first idea of **factorization** of a matrix. 

------


