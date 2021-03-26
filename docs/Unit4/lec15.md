# Generative Models

There are topics and exercises

## 1. Generative vs. Discriminative models

**Generative models** work by explicitly modelling the probability distribution of each of the individual classes in the training data. 

* e.g. **Gaussian generative models** fit a Gaussian probability distribution to the training data in order to estimate the probability of a new data point belonging to different classes during prediction.

**Discriminative models** learn explicit decision boundary between classes. 

* e.g. **SVM classifier** which is a discriminative model learns its decision boundary by maximizing the distance between training data points and a learned decision boundary.

## 2. Simple Multinomial Generative model

Consider a very simple multinomial model $M$ to generate one word $w$ at a time independently in a document with a fixed vocabulary $W$. So $M$ models the probability a word $w \in W$ is generated, which can be denote as $P(w | \theta ) = \theta _ w$, where $\theta_w$ is a parameter in our model $M$ indicating the probability of a model choosing the word $w$.

Note that since $\theta_w$ is a probability, $0 \le \theta _ w \le 1$ and $\sum _{w\in W} \theta _ w = 1$

## 3. Maximum Likelihood Estimate (MLE)

The **likelihood function** is the product of the probabilities of words from document $D$.
$$
p(D|\theta) = \prod^n_{i=1} \Theta_{w_i} = \prod_{w \in W} \theta_w^{\text{count}(w)}
$$
The **maximum likelihood estimate (MLE)**:
$$
\max_{\theta} P(D|\theta) = \max_{\theta} \prod_{w \in W} \theta_{w}^{\text{count}(w)}
$$
We take the log of it for computational convenience
$$
\text{log} \prod_{w \in W} \theta_w^{\text{count}(w)} = \sum_{w \in W} \text{count}(w) \text{log} \theta_w
$$
> #### Exercise 33
>
> Suppose a document contains two words $W_{\theta} = \{0,1\}$ where $\theta_0 = \theta, \theta_1 = 1 - \theta$. What is the MLE of model parameter？
>
> > **Answer**:  $\hat{\theta} = \frac{\text{count}(0)}{\text{count}(0) + \text{count}(1)}$
>
> > **Solution**: 
> >
> > The log of likelihood function is
> > $$
> > \sum_{w \in W} \text{count}(w) \text{log} \theta_w = \text{count}(0) \text{log} \theta + \text{count}(1) \text{log}(1-\theta)
> > $$
> > Take the derivative and set it to zero,
> > $$
> > \frac{\partial {(\text{count}(0) \text{log} \theta + \text{count}(1) \text{log}(1-\theta))}}{\partial \theta} = \frac{\text{count}(0)}{\theta} + \frac{\text{count}(1)}{1-\theta}= 0
> > $$
> >
> > We got
> > $$
> > \hat{\theta} = \frac{\text{count}(0)}{\text{count}(0) + \text{count}(1)}
> > $$
> >

> #### Exercise 34
>
> Suppose we us a multinomial model $M$ to generate documents that are English letter sequences $W = \{a,b,c,...,z\}$.  What is the minimal number of parameters that the model $M$ should have? 
>
> > **Answer**: 25
>
> > **Solution**: For a multinomial generative model we should have a parameter $\theta_w$ for each word $w \in W$, but since the parameters should sum up to one, we can express one of the parameters as 1 minus the sum of all others. 

> #### Exercise 35
>
> If we want to use MLE to fit the parameter $\lambda$ with the training data $x_1, x_2, ..., x_n$, what is the MLE for Poisson Distribution?
>
> The PMF of Poisson distribution is 
> $$
> P(X=x)=\frac{\lambda ^ x e^{-\lambda }}{x!}
> $$
>
> > **Answer**: $\frac{1}{n}\sum _ i x_ i$.
>
> > **Solution**: 
> >
> > Compute the log likelihood
> > $$
> > \begin{aligned}
> > \log \prod _ i P(X=x_ i) &= \log \prod _ i \frac{\lambda ^{x_ i} e^{-\lambda }}{x_ i!}\\
> > &= \sum _ i \log (\lambda ^{x_ i}) + \log (e^{-\lambda })-\log (x_ i!)\\
> > &= \log \lambda \sum _ i x_ i -n\lambda - \sum _ i \log (x_ i!)
> > \end{aligned}
> > $$
> > Take the derivative, set it to zero, and compute MLE for $\lambda$ 
> > $$
> > \frac{1}{\lambda }\sum _ i x_ i -n = 0\\
> > \lambda = \frac{1}{n}\sum _ i x_ i
> > $$
> > This is in accordance with the fact that $λ$ is the expectation of a Poisson variable with parameter $\lambda$.