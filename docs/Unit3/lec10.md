# Recurrent Neural Networks (RNNs)

There are 3 topics and 0 exercise.

## 1. Introduction

RNNs can be useful than ANNs because in general RNNs automatically address some issues that need to be engineered with feed-forward networks, such as answering "*How many time steps back should we look at in the feature vector?*"  

* An inconvenient aspect of feed-forward networks is that we have to **manually engineer how history is mapped to a feature vector (representation)**. This mapping into feature vectors (**encoding**) is also what we would like to learn. **RNN's learn the encoding into a feature vector**, unlike feed-forward networks.

Sentiment analysis, language translation, and language modelling can be realized by RNNs, but each **task requires a different sentence representation** as they focus on different parts of the sentence.

* Sentiment analysis focuses on the holistic meaning of a sentence, while translation focuses more on individual words.

## 2. Encoding with RNNs

This is a typical structure of a single-layered recurrent neural network. 

![rnns](../assets/images/U3/Lec10.jpg)
$$
s_ t = \tanh (W^{s,s}s_{t-1} + W^{s,x}x_ t)
$$
$s_t$ $(m \times 1)$ is new context or state, $s_{t-1}$ $(m \times 1)$ is current context or state. $x_t$ $(d \times 1)$ is new information. $W^{s,s}$ $(m \times m)$, $W^{s,x}$ $(m \times d)$ are parameters. $W^{s,s}$ and $W^{s,x}$ determine $\theta$, which can be adjusted. $W^{s,s}$ decides what part of the previous information to keep. $W^{s,w}$ takes into account new information.

Three **differences** between the encoder (unfolded RNN) and a standard feed-forward architecture.

* **Input is received at each layer** (per word), not just at the beginning as in a typical feed-forward network.
* The number of layers varies, and depends on the **length of the sentence**.
* Parameters of each layer (representing an application of an RNN) are **shared** (same RNN at each step).

One **problem** of RNN: 

* vanishing / exploding gradient

## 3. Gating and LSTM

#### Gating

The gate vector $g_t$ which is of the same dimension as $s_t$, determines "*how much information to overwrite in the next state*." In this case, it can learn to control how much to update.

A single-layered **gated RNN** can be written as
$$
\begin{aligned}
g_t & = \text {sigmoid}(W^{g,s}s_{t-1}+W^{g,x}x_{t})\\
s_t & = (1-g_ t) \bigodot s_{t-1} + g_ t \bigodot \tanh (W^{s,s}s_{t-1} + W^{s,x}x_ t).\\
\end{aligned}
$$
where the sign $⨀$ denotes element-wise multiplication.

#### LSTM

![rnns](../assets/images/U3/Lec10.jpg)

<img src="../assets/images/U3/Lec10_1.jpg" alt="lstm" style="zoom:67%; text-align:center;" />

$c_t$ represents the memory cell, and $h_t$ represents the visible state. $[c_t, h_t]$ represents the new context or state. $[c_{t-1}, h_{t-1}]$ represents the current context or state.

## 4. 

> Exercise 28
>
> 





