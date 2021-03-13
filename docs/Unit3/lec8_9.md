# Feedforward Neural Networks

There are 2 topics and 4 exercises.

## 1. Basics: Units & Layers

Given a neural network with one hidden layer for classification, the hidden layer is a **feature representation**, and the output layer is a **classifier** using the learned feature representation.

There're also other parameters that will affect the learning process and the performance of the model, such as the learning rate and parameters that control the network architecture (e.g. number of hidden units/layers) etc. These are often called **hyper-parameters**.

> #### Exercise 24:
>
> Which of the following is/are optimized during a single training pass? (Note that cross-validation is tuned before this point.) Check all that apply.
>
> A. The dimension of the feature representation
>
> B. The weights that control the feature representation
>
> C. The hyper-parameters
>
> D. The weights for the classifier
>
> > **Answer**: BD
>
> > **Solution**: 
> >
> > Neural network is similar to the linear classifier which learns the parameters for the classifier. However, in this case it also learns the parameters that generate a representation for the data.
> > The dimensions and the hyper-parameters are decided with the structure of the model and are not optimized directly during the learning process but can be chosen by performing a **grid search** with the evaluation data or by more advanced techniques (such as **meta-learning**).

## 2. Back Propagation & Stochastic Gradient Descent

The main steps in the stochastic gradient descent algorithm are demonstrated via training the simple neural network which is made up of $L$ hidden layers, each of which consists only one unit  / one activation function.

As usual, $x$ is the input, $z_i$ is the weighted combination of the inputs to the $i$th hidden layer. In this one-dimensional case, weighted combination reduces to products:
$$
\begin{aligned}
z_1 & = x w_1 \\
\text{ for }i = 2...L: z_i & = f_{i-1}w_i \text{ where }f_{i-1} = f(z_{i-1})\\
\end{aligned}
$$
The loss function can be 
$$
\mathcal{L}(y, f_ L) = (y - f_ L)^2
$$
where $y$ is the true value, and $f_L$ is the output of the neural network.

Since the value of a function is non-decreasing in the direction of its gradient from any given point in its domain, we update the weights in the direction opposite to that of the gradient from any point. The appropriate **stochastic gradient descent update rule** for the parameter $w_1$ with $\eta$ learning rate is
$$
w_1 \leftarrow w_1 - \eta \cdot \nabla _{w_1} \mathcal{L}(y, f_ L)
$$

> #### Exercise 25:
>
> Let $\delta _ i = \frac{\partial \mathcal{L}}{\partial z_ i}$, the first step to updating any weight $w$ is to calculate $\frac{\partial \mathcal{L}}{\partial w}$. What is the correct expression(s) for $\frac{\partial \mathcal{L}}{\partial w_1}$ ?
>
> > **Answer**: 
> > $$
> > \frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial z_1}{\partial w_1}\cdot \frac{\partial \mathcal{L}}{\partial z_1} \text{ OR} \\
> > \frac{\partial \mathcal{L}}{\partial w_1} = x\cdot \delta _1
> > $$

> #### Exercise 26:
>
> Assume that $f$ is the hyperbolic tangent function: 
> $$
> f(x) = tanh(x)\\
> f'(x) = (1-tanh^2(x))
> $$
> Which of the following option is the correct expression for $δ_1$ in terms of $δ_2$?
>
> > **Answer**: $\delta _1 = (1 - f_1^2)\cdot w_2\cdot \delta _2$
>
> > **Solution**:
> >
> > The chain rule gives 
> > $$
> > \delta _1 = \frac{\partial f_1}{\partial z_1}\cdot \frac{\partial z_2}{\partial f_1}\cdot \frac{\partial \mathcal{L}}{\partial z_2}.
> > $$
> > Since $f_1 = tanh(z_1),\,$ we have
> > $$
> > \frac{\partial f_1}{\partial z_1} = (1 - f_1^2).
> > $$
> > Since $z_2 = w_2\cdot f_1$, we have
> > $$
> > \frac{\partial z_2}{\partial f_1} = w_2.
> > $$
> > Substituting the values of $\frac{\partial f_1}{\partial z_1}, \frac{\partial z_2}{\partial f_1}$ into the main expression for $\delta _1$ we get:
> > $$
> > \delta _1 = (1 - f_1^2)\cdot w_2\cdot \frac{\partial \mathcal{L}}{\partial z_2} \, =\,  (1 - f_1^2)\cdot w_2\cdot \delta _2.
> > $$

> #### Exercise 27
>
> Now let the loss function to be $\mathcal{L}(y, f_ L) = (y - f_ L)^2.$ Compute $\frac{\partial \mathcal{L}}{\partial w_1}$ 
>
> > **Answer**: 
> > $$
> > \frac{\partial \mathcal{L}}{\partial w_1} = x (1-f_1^2)(1 - f_2^2)\cdots (1-f_{L}^2)w_2w_3\cdots w_ L(2(f_ L - y))
> > $$
>
> > **Solution**: 
> >
> > From Exercise 26 we know
> > $$
> > \frac{\partial \mathcal{L}}{\partial w_1} = x\cdot \delta _1\\
> > \delta _1 =\,  (1 - f_1^2)\cdot w_2\cdot \delta _2.
> > $$
> > Similarly, $\delta _2, \delta _3 ... \delta _ L$ can be given as follows:
> > $$
> > \delta_2 = (1 - f_2^2)\cdot w_3\cdot \delta _3\\
> > \delta_3 = (1 - f_3^2)\cdot w_4\cdot \delta _4\\
> > \vdots\\
> > \delta_{L-1} = (1 - f_{L-1}^2)\cdot w_ L\cdot \delta _ L\\
> > $$
> > and 
> > $$
> > \begin{aligned}
> > \delta_L & = \frac{\partial \mathcal{L}}{\partial f_ L}\cdot \frac{\partial f_ L}{\partial z_ L}\\
> > &  = \frac{\partial (f_ L - y)^2}{\partial f_ L} \frac{\partial f_ L}{\partial z_ L}\\
> > &  = 2 (f_ L - y) \frac{\partial f_ L}{\partial z_ L}\\
> > &  = 2 (f_ L - y) (1 - f_ L^2).
> > \end{aligned}
> > $$
> > Plugging the above equations into the expression for $\frac{\partial \mathcal{L}}{\partial w_1}$ we get:
> > $$
> > \begin{aligned}
> > \frac{\partial \mathcal{L}}{\partial w_1} & = x\cdot \delta _1\\
> > & = x\cdot (1 - f_1^2)\cdot w_2\cdot \delta _2\\
> > & = x\cdot (1 - f_1^2)\cdot w_2\cdot (1 - f_2^2)\cdot w_3\cdot \delta _3\\
> > &~~ \vdots\\
> > &  = x (1-f_1^2)(1 - f_2^2)\cdots (1-f_{L}^2)w_2w_3\cdots w_ L(2(f_ L - y))\\
> > \end{aligned}
> > $$