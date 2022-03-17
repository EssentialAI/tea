#!/usr/bin/env python
# coding: utf-8

# # [2.1] Infinite Series

# *Gentle Introduction to Probability*
# 
# Imagine that you have a <span class = 'high'>fair coin</span>. If you get a tail, you flip the coin again. You do this repeatedly until you finally get a head. <span class = 'high'>What is the probability that you need to flip the coin three times to get one head?</span>
# 
# This is a warm-up exercise. Since the coin is fair, the probability of obtaining a head is $\frac{1}{2}$. The probability of getting a tail followed by a head is $\frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$. If you follow this logic, you can write down the probabilities for all other cases. The below figures shows these probabilities for better understanding.
# 
# ![test](../images/coins.png)

# We can also summarize these probabilities using <span class = 'high'>Histogram</span> as shown below. We see that the sequence above can be infinitely long.

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
p = 1/2
n = np.arange(0,10)
X = np.power(p,n)
plt.bar(n,X)
plt.xlabel("Number of coin flips")
plt.ylabel("Probability of head on last flip")
plt.show()


# Another question that can be asked based on the above experiment is, <span class = 'high'>On average, if you want to be $90\%$ sure that you will get a head, what is the minimum number of attempts you need to try?</span>
# 
# This problem can be answered by analyzing the sequence of probabilities. If we make two attempts, then the probability of getting a head is the sum of the probabilities for one attempt and that of two attempts:
# 
# $$
# \begin{align}
# \mathbb{P}[\text{success after 1 attempt}] &= \frac{1}{2} = 0.5 \\
# \mathbb{P}[\text{success after 2 attempts}] &= \frac{1}{2} + \frac{1}{4}= 0.75 \\
# \mathbb{P}[\text{success after 3 attempts}] &= \frac{1}{2} + \frac{1}{4} + \frac{1}{8} = 0.875 \\
# \mathbb{P}[\text{success after 4 attempts}] &= \frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{16}= 0.9375
# \end{align}
# $$
# 
# This means if we try for 4 attempts we will have $93.75 \%$ probability to obtain a head.
# 
# This section is a gentle introduction to calculation of probability without formulae. Probability will be reviewed in detail in the next chapter.

# #### [2.1.1] Geometric Series
# 
# A geometric series is the sum of a finite or an infinite sequence of numbers with a constant ratio between successive terms. As we have seen in the previous example, a geometric series appears naturally in the context of discrete events. In <span class = 'high'>chapter</span>, we will use geometric series when calculating <span class = 'high'>expectation</span> and <span class = 'high'>moments</span> of a random variable.
# 
# ```{admonition} Geometric Series
# :class: note
# <span class = 'high'>Finite Geometric Sequence</span> of power $n \enspace \rightarrow \enspace \{1,r,r^2,...,r^n\}$
# 
# <span class = 'high'>Infinite Geometric Sequence</span> of numbers $\enspace \rightarrow \enspace \{1,r,r^2,...\}$
# ```
# 
# ```{admonition} Sum of geometric series
# :class: note
# Sum of a finite geometric series of power $n$ is:
# 
# 
# $$\sum_{k=0}^n r^k = 1+r+r^2+...+r^n = \frac{1-r^{n+1}}{1-r} \label{naresh}$$
# 
# Sum of an infinite geometric series is (if $0 < r < 1$):
# 
# 
# ```{math}
#     \sum_{k=0}^\infty r^k = 1+r+r^2+... = \frac{1}{1-r}
# ```
# 
# $$\sum_{k=1}^{\infty}k r^{k-1} = 1+2r+3r^2+... = \frac{1}{(1-r)^2}$$
# :::{seealso}
# $$
# \sum_{n=1}^\infty \frac{1}{2^k} = \frac{1}{4}(1+\frac{1}{2}+\frac{1}{4}...) = \frac{1}{4}\cdot \frac{1}{1-\frac{1}{2}} = \frac{1}{2}$$
# 
# $$\sum_{n=1}^\infty \frac{1}{n^2} = 1+\frac{1}{2^2}+\frac{1}{3^2}+.. = \frac{\pi^2}{6}
# $$
# :::
# ```
# Proofs for above equations to be added later.

# Trying to refer {eq}`(2.1.1)`
