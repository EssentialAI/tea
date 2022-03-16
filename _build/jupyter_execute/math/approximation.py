#!/usr/bin/env python
# coding: utf-8

# # [2.3] Approximation

# Consider a function $f(x) = log(1+x)$, for $x >0$. This is a non-linear function. Non-linear functions are hard to deal with because, for example, if you want to integrate the function $\int_{a}^{b}xlog(1+x)dx$, then the logarithm will force us to do integration by parts. However, in many practical problems, we may not need the full range of $x>0$. Suppose that you are only interested in the values $x << 1$. Then the logarithm can be approximated, and thus the integral can also be approximated

# In[23]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(6)
y = np.log(1+x)

fig,(ax1, ax2) = plt.subplots(1, 2,figsize=(12,5))
fig.suptitle('The function f(x)=log(1+x) and the approximation $\hatf(x)=x$')
ax1.plot(x, y,label='f(x)=log(1+x)')
ax1.plot(x,x, label = 'f(x)=x')
ax1.set_ylim([0, 2])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

ax2.plot(x, y, label = 'f(x)=log(1+x)')
ax2.plot(x,x, label = 'f(x)=x')
ax2.set_ylim([0, 0.2])
ax2.set_xlim([0, 0.2])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
plt.show()


# To see how this is possible, let's see the above image. The non-linear function $f(x) = log(1+x)$ and an approximation $\hat f(x)=x$ are chosen. The approximation is carefully chosen such that for $x<<1$, the approximation $\hat f(x)$ is close to the true function $f(x)$. Therefore, we can argue that for $x<<1$,
# 
# $$log(1+x) \approx x$$
# 
# This simplifies the calculation for the integral discussed above. For example, if you want to integrate $xlog(1+x)$ for $0<x<0.1$, then the integral can be approximated by,
# 
# $$\int_0^{0.1}xlog(1+x)dx \enspace \approx \int_0^{0.1}x^2dx = \frac{x^3}{3} = 3.33 \times 10^{-4}$$
# 
# (The actual integral is $3.21 \times 10^{-4}$). In this section we will learn about the basic approximation techniques.

# ### [2.3.1] Taylor Approximation
# 
# Given a function $f: \mathbb{R} \rightarrow \mathbb{R}$, it is often useful to analyze its behaviour by approximating $f$ using its local information. <span class = 'high'>Taylor approximation</span> is one of the tools for such a task. We will use the Taylor approximation on many occasions.
# 
# ```{admonition} Taylor Approximation
# Let $f: \mathbb{R} \rightarrow \mathbb{R}$ be a continuous function with infinite derivatives. Let $a \in \mathbb{R}$ be a fixed constant. The Taylor approximation of $f$ at $x=a$ is
# 
# $$
# \begin{align}
# f(x) &= f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+... \\
# &= \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n
# \end{align}$$
# $\small{\text{where }f^{(n)} \text{ denotes the } n^{th} \text{ -order derivative of } f}$
# ```

# Taylor approximation is a geometry-based approximation. It approximated the function according to the offset, slope, curvature, and so on. According to the definition the Taylor series has an infinite number of terms. If we use a finite number of terms, we obtain the $n^{th}$-order Taylor approximation.
# 
# $$
# \text{First-Order: } f(x) = \underbrace{f(a)}_{\text{offset}}+\underbrace{f'(a)(x-a)}_{\text{slope}}+ \mathcal{O}((x-a)^2)$$
# 
# $$\text{Second-Order: } f(x) = \underbrace{f(a)}_{\text{offset}}+\underbrace{f'(a)(x-a)}_{\text{slope}}+ \underbrace{\frac{f''(a)}{2!}(x-a)^2}_{\text{curvature}}+\mathcal{O}((x-a)^3)
# $$
# 
# Here, the big-O notation $\mathcal{O}(\epsilon^k)$ means any term that has an order at least power $k$. For small $\epsilon$, i.e., $\epsilon ≪ 1$, a high-order term $\mathcal{O}(\epsilon^k) \approx 0$ for large $k$.
