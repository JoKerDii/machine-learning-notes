# Convolutional Neural Network

There are 2 topics.

## Convolution

#### Continuous Convolution

The **convolution** is an operation between 2 functions $f$ and $g$.
$$
(f * g)(t) \equiv \int _{-\infty }^{+\infty } f(\tau )g(t-\tau )d\tau
$$
In this integral, $\tau$ is the **dummy variable** for integration and $t$ is the **parameter**. Intuitively, convolution 'blends' the two function $f$ and $g$ by expressing the amount of overlap of one function as it is shifted over another function.
The **area** under the convolution can be computed as
$$
\begin{aligned}
\int _{-\infty }^{+\infty } (f * g) dt & = \int _{-\infty }^{+\infty } [\int _{-\infty }^{+\infty } f(\tau )g(t-\tau )d\tau ] dt \\
& = \int _{-\infty }^{+\infty } f(\tau ) [\int _{-\infty }^{+\infty } g(t-\tau )dt] d\tau \\ 
& = [\int _{-\infty }^{+\infty } f(\tau ) d\tau ][\int _{-\infty }^{+\infty } g(t) dt]
\end{aligned}
$$
which is the product of the areas under $f$ and $g$.

#### 1D Discrete Convolution

The convolution is defined as
$$
(f * g)[n] \equiv \sum _{m = -\infty }^{m = +\infty } f[m]g[n-m]
$$

## Convolution and Cross-Correlation

Let **signal** to be $f(t)$, **filter** to be $g(t)$, **convolution** is defined as:
$$
(f * g)(t) = \int^{+\infty}_{-\infty}f(t)g(-\tau + t) dt
$$
**Cross-correlation** is defined as
$$
(f * g)(t) = \int^{+\infty}_{-\infty}f(t)g(\tau + t) dt
$$
The difference is whether the filter $g(t)$ is reversed. But convolution and cross-correlation are the same no matter whether it is reversed or not. Since the result $f * g$ is also reversed, and the areas under the curve are the same. In practice, cross-correlation is more favorable than convolution since it does not require $g(t)$ to be reversed and thus computationally simple.



