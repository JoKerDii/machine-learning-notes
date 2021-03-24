## Unsupervised Learning

There are topics and exercises

## Clustering

A **partition** of a set is a grouping of the set's elements into non-empty subsets, in such a way that **every** element is included in one and only one of the subsets. In other words, $C_1,C_2,...,C_K$ is a partition of $\{1,2,...,n\}$ if and only if
$$
C_1 \cup C_2 \cup ... \cup C_ K = \big \{  1, 2, ..., n \big \}
$$
and
$$
C_ i \cap C_ j = \emptyset \quad \text {for any $i \neq j$ in $\big \{ 1, ..., k\big \} $ }
$$
**Input** of clustering: 

* Set of feature vectors $S_ n = \big \{ x^{(i)} | i =1,...,n \big \}$
* The number of clusters $K$
* The representatives of each cluster $z_1, ..., z_K$

**Output** of clustering

* A partition of **indices** $\big \{ 1, ..., n\big \}$ into $K$ sets, $C_1, ..., C_K$.
* "Representatives" in each of the $K$ partition sets, given as $z_1, ..., z_K$

## Similarity Measures

* Cosine distance
  $$
  cos(x^{i}, x^{j}) = \frac{x^{i} \cdot x^{j}}{\|x^{i}\| \cdot \|x^{j}\|}
  $$

* Euclidean distance
  $$
  \text{dist}(x^{i}, x^{j}) = \|x^{i} - x^{j}\|^2
  $$

* 

> #### Exercise 29
>
> If we want to measure the similarity between two Google News articles, suppose you assume that the length of an article does not tell any useful information about the article, and hence choose a similarity measure that does not depend on the length of the article. Which of the following similarity measure could be the one you chose?
>
> A. Euclidean distance
>
> B. Cosine distance
>
> > **Answer**: B
>
> > **Solution**: It can be thought that **longer articles will have larger norms**, since they are more likely to contain unique words. Because it is assumed that the length of the article does not contain any important information, it is not ideal to use the Euclidean distance.

## Cost Functions

The total cost of clustering output is defined as the sum of the cost inside each cluster. 
$$
\text {Cost}(C_1, ..., C_ K) = \sum _{j=1}^{K} \text {Cost}(C_ j)
$$
where $\text {Cost}(C_ j)$ is supposed to measure "how homogeneous" the assigned data are inside the $j$th cluster $C_j$.

If $\text {Cost}(C_ j)$ is **sum of the squared Euclidean distance**, the cost the clustering output is 
$$
\text {Cost}(C_1, ..., C_ K) = \sum _{j=1}^{K} \text {Cost}(C_ j) = \sum _{j=1}^{K} \sum _{i \in C_ j} \| x_ i - z_ j\| ^2
$$


