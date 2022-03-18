#!/usr/bin/env python
# coding: utf-8

# # [2.2] Binomial Series

# A [geometric series](math_back1.md) is useful when handling situations such as $N-1$ failures followed by a success. However, we can easily twist the problem by asking: <span class = 'high'>What is the probability of getting one head out of $3$ independent coin tosses?</span> In this case, the probability can be determined by enumerating all possible cases:
# 
# $$
# \begin{align}
# \mathbb{P}[\text{1 head in 3 coins}] &= \mathbb{P}[H,T,T]+\mathbb{P}[T,H,T]+\mathbb{P}[T,T,H] \\
# &= \small{\left(\frac{1}{2}\times \frac{1}{2} \times \frac{1}{2}\right)+\left(\frac{1}{2}\times \frac{1}{2} \times \frac{1}{2}\right)+\left(\frac{1}{2}\times \frac{1}{2} \times \frac{1}{2}\right)} \\
# &= \small{\frac{3}{8}}
# \end{align}
# $$
# 
# ![test](../images/coins1.png)

# In general, the number of combinations can be systematically studied using <span class = 'high'>Combinatorics</span>, which we will discuss later in the chapter. However, the number of combinations motivates us to discuss another background technique known as the <span class = 'high'>Binomial Series</span>. The binomial series is instrumental in algebra when handling polynomials such as $(a+b)^2$ or $(1+x)^3$. It provides valuable formula when computing these powers.
# 
# ```{admonition} Binomial Theorem
# :class: note
# For any real numbers $a$ and $b$, the binomial series of power $n$ is:
# 
# 
# $$\definecolor{orange}{RGB}{232, 62, 140}
# \color{orange}{(a+b)^n = \sum_{k=0}^{n}\binom{n}{k}a^{n-k}b^k}$$
# 
# $$
# \small{\text{where} \binom{n}{k} = \frac{n!}{k!(n-k)!}}$$
# ```

# In[49]:


from scipy.special import comb, factorial
n=10
k=2
print(comb(n,k))
print(factorial(k))


# ```{admonition} Pascal's Identity
# Let $n$ and $k$ be positive integers such that $K \leq n$, then,
# 
# $$\binom{n}{k}+\binom{n}{k-1}=\binom{n+1}{k}$$
# ```

# ```{dropdown} Exercise 1: Using Binomial theorem find $(1+x)^3$
# 
# $$\begin{align}
# (1+x)^3 &= \sum_{k=0}^{n}\binom{3}{k}1^{3-k}x^k \\
# &= 1+3x+3x^2+x^3
# \end{align}$$
# ```
