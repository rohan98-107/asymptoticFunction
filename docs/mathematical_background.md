# Mathematical Background 

This document summarizes the mathematical background required to understand the objects computed in the 'asymptoticFunction' package. We do not prove anything - we simply introduce the notions that motivate the creation of the package. 
We will work exclusively in finite dimensional Euclidean space $\mathbb{R}^n$ with norm $\| \cdot \|$. This document assumes knowledge of basic real analysis. 

## Asymptotic Directions 

An asymptotic direction is simply a vector $d \in \mathbb{R}^n$, which is typically normalized, that represents the overall direction of an unbounded sequence. 
Formally: for a sequence of vectors $\{x_k\} \subseteq \mathbb{R}^n$, if there exists a sequence of positive scalars $\{t_k\} \subseteq \mathbb{R}^+$ and 
$$ \lim_{k \to \infty} \frac{x_k}{t_k} = d, $$
then $d$ is called an asymptotic direction. We make two comments: first it is often convenient to take $t_k := \|x_k\|$ so that the direction $d$ is a normalized vector. Second take a constant sequence $x_k = x \in \mathbb{R}^n$ then one can take any indexed sequence of scalars such as $t_k := k$ and obtain the limit zero. Thus the zero vector is trivially an asymptotic direction. 

<div align="center">
  <figure>
    <img src="figures/asymptotic_direction.svg" width="70%">
    <figcaption>
      An unbounded sequence with increasing transverse oscillations and its
      asymptotic direction, shown as the bold red ray.
    </figcaption>
  </figure>
</div>


## Asymptotic Functions 

Let $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ be a proper function i.e., let $f$ not take the value $-\infty$ and let $f(x) < +\infty$ for some $x \in \mathbb{R}^n$. Its asymptotic function $f_\infty: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 
is given by 
$$ f_\infty(d) = \liminf_{\substack{ t \to \infty \\ d' \to d }} \frac{f(td')}{t}. $$
The asymptotic function describes the behavior of $f$ in the direction $d$. In particular $f_\infty(d) < 0$ indicates descent along the ray $\lambda d$ for any $\lambda > 0$ while $f_\infty(d) > 0$ indicates growth along $\lambda d$. 

<div align="center">
  <figure>
    <img src="figures/asymptotic_function.svg" width="70%">
    <figcaption>
      An unbounded sequence with increasing transverse oscillations and its
      asymptotic direction, shown as the bold red ray.
    </figcaption>
  </figure>
</div>

## Asymptotic Cones

Consider a closed set $X \subseteq \mathbb{R}^n$. The asymptotic cone of the set $X_\infty \subseteq \mathbb{R}^n$ is the set of asymptotic directions of the set $X.$ That is, for any sequence of the form $\{x_k\} \subseteq X$, if there exist positive scalars $\{t_k\} \subseteq \mathbb{R}^+$ for which $x_k/t_k \to d$ then $d \in X_\infty$. 
It is convenient to choose $t_k := \|x_k\|$ as this does not affect the size of the cone $X_\infty$. In the 'asymptoticFunction' package we compute an approximation of the asymptotic cone only for sets of the kind 

$$ 
\begin{aligned}
X &= \{ x \in \mathbb{R}^n \mid f_i(x) \le 0, g_j(x) = 0 \text{ for } i \in \mathcal{I}, j \in \mathcal{J} \} \\
&= \left( \bigcap_{i \in \mathcal{I}} \{ x \mid f_i(x) \leq 0 \} \right) \cap \left( \bigcap_{ j \in \mathcal{J} } \{x \mid g_j(x) = 0 \} \right)
\end{aligned}
$$ 


where each $f_i, g_j$ are proper and $\mathcal{I}, \mathcal{J}$ are finite index sets. In all package documentation that follows we denote this approximation by 

$$ 
\begin{aligned}
\overline{X}_\infty &:= \{ d \in \mathbb{R}^n \mid (f_i)_\infty(d) \le 0, (g_j)_\infty(d) = 0 \text{ for } i \in \mathcal{I}, j \in \mathcal{J} \} \\
&= \left( \bigcap_{i \in \mathcal{I}} \{ d \mid (f_i)_\infty(d) \leq 0 \} \right) \cap \left( \bigcap_{ j \in \mathcal{J} } \{d \mid (g_j)_\infty(d) = 0 \} \right)
\end{aligned}
$$
<div align="center">
  <figure>
    <img src="figures/asymptotic_cone.svg" width="70%">
    <figcaption>
      An unbounded sequence with increasing transverse oscillations and its
      asymptotic direction, shown as the bold red ray.
    </figcaption>
  </figure>
</div>

If every function $f_i, g_j$ were assumed to be convex then $X_\infty = \overline{X}_\infty$. Without this assumption we have simply the inclusion $X_\infty \subseteq \overline{X}_\infty$. 
