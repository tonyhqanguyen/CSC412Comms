# The Gumbel-Softmax Reparametrization Trick for Categorical Sampling in Stochastic Neural Networks

## A motivating example
MolGAN is a generative model for representing molecular compounds using graphs. It has the GAN architecture as proposed by Goodfellow et. al. Each node in the generated graph is drawn from a categorical distribution of atom types. During the forward pass of this network, the model outputs the sampled atoms. To optimize the network weights, the model propagates backwards through the layers to compute the necessary gradients. However, gradients cannot be computed for stochastic or discrete operations. Hence, when optimizing a model like MolGAN, we need to do more than simply draw samples from the distribution. 

## Undifferentiable operations
Gradients cannot be computed for stochastic nodes. In more detail, when we compute loss of a model like MolGAN, we generate some sample(s) $z \sim p(z)$ and compute $\mathbb{E}_{p(z)}[f_\theta(z)]$. Here, $z$ is distributed according to $p(z)$, $f_\theta(z)$ is the neural network which we pass the generated samples through. Therefore, when computing the gradients with respect to the parameters $\theta$ of the network, we compute:
$$\begin{aligned}
  \nabla_\theta \mathbb{E}_{p(z)}[f_\theta(z)] &= \nabla_\theta\left[\int_z p(z)f_\theta(z)dz\right]\\
                                 &= \int_z p(z)\left[\nabla_\theta f_\theta(z)\right]dz \tag{1}\\
                                 &= \mathbb{E}_{p(z)}[\nabla_\theta f_\theta(z)]
\end{aligned}$$

It is observable here that the gradient w.r.t. $\theta$ of the expectation is equal to the expectation of gradient w.r.t. $\theta$. However, in the case of MolGAN or other stochastic networks like the VAE, part of the set of parameters $\theta$ includes the parameters to the distribution $p(z)$. Hence this distribution should be written as $p_\theta(z)$ as it is parameterized by $\theta$. In a situation like this, a problem arises:
$$\begin{aligned}
  \nabla_\theta \mathbb{E}_{p_\theta(z)}[f_\theta(z)] &= \nabla_\theta\left[\int_z p_\theta(z)f_\theta(z)dz\right]\\
                                 &= \int_z \nabla_\theta\left[p_\theta(z)f_\theta(z)\right]dz\\
                                 &= \int_z f_\theta(z)\nabla_\theta p_\theta(z)dz + \int_z p_\theta(z)\nabla_\theta f_\theta(z)dz \tag{2}\\
                                 &= \int_z f_\theta(z)\nabla_\theta p_\theta(z)dz + \mathbb{E}_{p_\theta(z)}[\nabla_\theta f_\theta(z)]
\end{aligned}$$

From the last line of $(2)$, the familiar expectation of the gradient from $(1)$ resurfaces. However, examining the first term on the last line of $(2)$, the computation $\nabla_\theta p_\theta(z)$ is not necessarily feasible. Here, we take advantage of Monte Carlo methods to estimate the expectation. Such methods only require that the distribution in use must be able to be sampled from, but not necessarily differentiable. 

## Reparametrization and the Gumbel-Max Trick
A common solution to overcome the issue of not being to differentiate stochastic nodes is to reparametrize the network. Instead of having the network's output being realizations of a random variable, the reparametrization trick expresses the output to be a differentiable transformation of a stochastic element $\epsilon$. For instance, instead of having the network output a realization $z$ from the standard Gaussian distribution $\mathcal{N}(0, 1)$, we can reparametrize $z = \mu + \sigma \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$. We would then perform backpropagation to learn $\mu$ and $\sigma$.<br/><br/>

<center><img src="Images/reparameterization.png"/></center><br/>

In the figure above, the left network directly outputs $f(z)$ where $z$ is a randomly sampled variable distributed with mean $\mu$ and variance $\sigma^2$ (denoted as a set of parameters $\phi$ in the image). In the network to the right, $z$ is expressed as a transformation of $\phi$ and random noise $\epsilon$. The red circular nodes are stochastic in their respective networks. In the left network, $z$ is a stochastic node, and backpropagating through this node is not possible due to the inability to differentiate the random sampling process. In comparison, the right network expresses $z$ as a deterministic and differentiable transformation of earlier nodes, hence $z$ is no longer stochastic. Therefore, backpropagation can be performed through $z$ to learn the parameters $\phi$. This technique is widely used in machine learning tasks like variational inference or to train stochastic networks like VAEs. 

This reparametrization trick solves the issue of stochastic nodes being unable to be backpropagated through due to them not being differentiable. However, stochastic processes are not the only ones that pose hindrances for network training algorithms. Specifically, random sampling from categorical distributions is not differentiable, as well. 

Let $C$ be a random variable distributed categorically with class probabilities $\pi_1, \pi_2, ..., \pi_n$. Then a common procedure for sampling $C$ is:
  1. Sample $U \sim Uniform(0, 1)$
  2. Return $i$ such that: $\verb|cumsum(i)| = \sum_{k = 1}^i \pi_k \leq U$ and $\forall j, \verb|cumsum(j)| \leq U \implies j \leq i$.

Essentially, we compute $\displaystyle\max_i \left\{\displaystyle\sum_{k = 1}^i \pi_k \leq U\right\}$ where $U \sim Uniform(0, 1)$. Here, the returned $i$ is the class index of the categorical distribution. 

The $max$ operation is not differentiable. Hence, if the sampling is done in this manner in a network like MolGAN, it would not be possible to backpropagate through the layers and compute the gradients to optimize the parameters. Thus, the reparametrization trick cannot be applied here, because the class probabilities of a categorical distribution cannot be explicitly used a differentiable transformation to create a deterministic output that can be backpropagated through.

This is where the Gumbel-Max trick comes into play. Introduced by Maddison et. al., this trick allows for the reparametrization trick to be applied to discrete distributions. Using the same categorical variable as above, $C \sim Categorical(\pi_1, \pi_2, ..., \pi_n)$, we draw a sample $z = \displaystyle\argmax_i \left\{\log\pi_i + G_i\right\}$, where $G_1, G_2, ..., G_n$ are identically and independently distributed as $Gumbel(0, 1)$. 

Let's show that by doing this, that $P(C = k)$, the probability of drawing the $k$-th category is equal exactly to $\pi_k$. Because we are using the standard Gumbel distribution, we have $P(G_i \leq g) = \exp\{-\exp\{-g\}\}$. Using this information, we will inspect the case where the $k$-th category is drawn by following the procedure as described above. In this situation, it is possible to compute the probability that $\forall k' \in \{1, 2, ..., n\}, z_{k'}\leq z_k$. In the following derivation, we will denote this event as $K$. Recall here that $z_i = \log\pi_i + G_i$. 

$$\begin{aligned}
P(K |z_k, \{\pi_{k'}\}) &= \prod_{k' \neq k} P(z_k \geq z_k') \\
&= \prod_{k' \neq k} P(z_k - z_{k'} \geq 0) \\
&= \prod_{k' \neq k} P(z_k - \log\pi_{k'} - G_{k'} \geq 0) \\
&= \prod_{k' \neq k} P(G_{k'} \geq z_k - \log\pi_{k'}) \\
&= \prod_{k' \neq k} \exp\{-\exp\{-(z_k - \log\pi_{k'})\}\}
\end{aligned}$$

The expression above is the probability that a _specific_ $z_k$ is the largest amongst all the $z_1, ..., z_n$. Next, we need to marginalize over $z_k$ as we want to find the general probability that the $k$-th category is selected, for _any_ $z_k$ value. In the following derivation, we use the fact that the probability density function for a standard Gumble variable $G_i$ at value $g$ is $\exp\{-g + \exp\{-g\}\}$. So we want to compute: 
$$\begin{aligned}
P(K | \{\pi_{k'}\}) &= \int_{z_k} P(z_k)P(K | z_k, \{\pi_{k'}\})dz_k\\
&= \int_{z_k} \left[\exp\{-(z_k - \log\pi_k) - \exp\{-(z_k - \log\pi_k)\}\} \cdot \prod_{k' \neq k} \exp\{-\exp\{-(z_k - \log\pi_{k'})\}\} \right]dz_k\\
&= \int_{z_k} \left[\exp\{-z_k + \log\pi_k - \exp\{-(z_k - \log\pi_k)\} + \sum_{k \neq k'} -\exp\{-(z_k - \log\pi_{k'})\}\}\right]dz_k \\
&= \int_{z_k} \left[\exp\{-z_k + \log\pi_k - \sum_{i = 1}^n \exp\{-(z_k - \log\pi_{i})\}\}\right]dz_k\\
&= \int_{z_k} \left[\exp\{-z_k + \log\pi_k - \sum_{i = 1}^n \exp\{-z_k\}\exp\{\log\pi_{i}\}\}\right]dz_k\\
&= \int_{z_k} \left[\exp\{-z_k + \log\pi_k -\exp\{-z_k\}\sum_{i = 1}^n \pi_i\}\right]dz_k\\
&= \int_{z_k} \left[\exp\{-z_k + \log\pi_k -\exp\{-z_k\}\}\right]dz_k\\
&= \int_{z_k} \exp\{-z_k\}\cdot\pi_k\cdot\exp\{-\exp\{-z_k\}\}dz_k \tag{3}\\
\end{aligned}$$

To compute this integration, we notice the following. Let $U \sim Uniform(0, 1)$, $Y = -\log\log\displaystyle\frac{1}{U}$. Then $Y \sim Gumbel(0, 1)$. Therefore $U \in [0, 1] \implies Y \in (-\infty, \infty)$. We proceed to compute the above integral.

Let $u = -\exp\{-z_k\}$. Then $du = \exp\{-z_k\}dz_k \iff \displaystyle\frac{1}{\exp\{-z_k\}}du = dz_k$. Using this substitution into the last line of $(3)$, we have:

$$\begin{aligned}
P(K | \{\pi_{k'}\}) &= \pi_k \int_{-\infty}^\infty \exp\{u\}du\\
&= \pi_k\cdot \displaystyle\lim_{R \rightarrow \infty} \bigg[\exp\{-\exp\{-z_k\}\}\bigg]^R_{-R} \\
&= \pi_k \cdot \displaystyle\lim_{R \rightarrow \infty} \left[\frac{1}{\exp\{\exp\{-R\}\}} - \frac{1}{\exp\{\exp\{R\}\}}\right]\\
&= \pi_k
\end{aligned}$$

Hence, by using the Gumbel-Max trick, the probability of drawing a class from a discrete distribution is equal to the class' probability. We are not done here, though. From calculus, it has been proven that a differentiable function is continuous. Therefore, using the contrapositive, a function that is not continuous is not differentiable. Obviously, the $\argmax$ function used in the Gumbel-Max trick to sample from discrete distributions is not continuous, thus not differentiable. Therefore, this trick is not sufficient for training a stochastic neural network using backpropagation.

## Gumbel-Softmax Approximation
The problem of not being to differentiate the $\argmax$ function can be circumvented by applying a softmax to the computed $z_k$. Using the Gumbel-Softmax approximation, instead of drawing discrete samples, usually encoded using one-hot vectors, we might draw interpolations of the classes. In other words, we draw a sample that is represented by a vector, where each coordinate of the vector shows the strength of the corresponding class in the interpolation. The drawn vector $y$ is computed using the formula:

$$y_i = \frac{\displaystyle\exp\left\{\frac{\log(\pi_i) + g_i}{\tau}\right\}}{\displaystyle\sum_{j = 1}^n \displaystyle\exp\left\{\frac{\log(\pi_j) + g_j}{\tau}\right\}} \ \ \ \forall i = 1, 2, ..., n$$

Here, $g_i$ is the realization of the standard Gumbel variable $G_i \sim Gumbel(0, 1)$. $\tau$ is a parameter of the distribution, referred to as the "temperature". It controls how closely the approximation models the categorical distribution of interest. From the formulation above, it can be seen that as $\tau \rightarrow 0$, the approximation becomes an $\argmax$. In this case, the largest $\exp\{\log(\pi_i) + g_i\}$ would be divided by a value very close to $0$, and hence would explode in magnitude. Therefore, the corresponding $y_i$ would have value very close to 1, making this approximation essentially calculate an $\argmax$. In the opposite extreme, as $\tau \rightarrow \infty$, the approximation becomes a uniform distribution over the categories.

To use this approximation technique in training a stochastic neural network, multiple words that incorporate this technique uses an annealing schedule to reduce the temperature over time. In other words, in earlier training iterations, the parameter $\tau$ is set to some value significantly greater than 0. Over the course of the training, $\tau$ is annealed to become closer and closer to 0, but never exactly 0.

Below is a widget that can be used to visualize the Gumbel-Softmax approximation of a categorical distribution with 5 classes. Slide the $\tau$ bar to adjust the temperature. Click the "sample" button to re-sample using the current parameters, by doing this, the class probabilities will also change.
