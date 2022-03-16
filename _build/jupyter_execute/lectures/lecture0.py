#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra Review
# 
# Linear algebra provides a way of compactly representing and operating on sets of linear equations. For example:
# 
# $$
# \begin{align}
# 4x_{1}-5x_{2} &= -13 \\
# -2x_{1}+3x_{2} &= 9
# \end{align}
# $$
# 
# <p style="text-align:center">The matrix notation of above equations is:</p>
# 
# $$Ax =b$$
# 
# $$\text{with } A = \begin{bmatrix}
#       4 & -5 \\
#       -2 & 3
#       \end{bmatrix}, \enspace b = \begin{bmatrix}
#       -13 \\
#       9
#       \end{bmatrix}$$
# 
# By $A \in \mathbb{R}^{m \times n}$, we denote a matrix with $m$ rows and $n$ columns. By $x \in \mathbb{R}^n$, we denote vector with $n$ entries.
# 
# <br>
# 
# $$\begin{align}A = \begin{bmatrix}
#       a_{11} & a_{12} & a_{13} & ... & a_{1n} \\
#       a_{21} & a_{22} & a_{23} & ... & a_{2n} \\
#       a_{31} & a_{32} & a_{33} & ... & a_{3n} \\
#       ... & ... & ... & ... & ... \\
#       a_{m1} & a_{m2} & a_{m3} & ... & a_{mn}
#       \end{bmatrix}, \enspace \enspace x = \begin{bmatrix}
#       x_{1} \\
#       x_{2} \\
#       x_{3} \\
#       .. \\
#       x_{n}
#       \end{bmatrix}
#       \end{align}$$
# 
# <br>
# 
# We denote the $j$th column of $A$ by $a^j$ or $A_{:,j}$:
# 
# $$A = \begin{bmatrix}
#       | & | & | & ... & | \\
#       a^1 & a^2 & a^3 & ... & a^n \\
#       | & | & | & ... &|
#       \end{bmatrix}$$
# 
# <br>
#       
# We denote the $i$th row of $A$ by $a^T$ or $A_{i,:}$:
# 
# $$A = \begin{bmatrix}
#       - a_{1}^T - \\
#       - a_{2}^T - \\
#       .. \\
#       - a_{m}^T - 
#       \end{bmatrix}$$
#       
# <br>      
# 
# ## 1. Matrix Multiplication
# 
# The product of two matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$ is the matrix
# 
# $$
# C = AB \in \mathbb{R}^{m \times p} \enspace \text{where, } C_{ij} = \sum_{k=1}^{n}A_{ik}B_{kj}$$
# 
# Note that in order for the matrix product to exist, the number of columns in $A$ must equal the number of rows in $B$.
# 
# ### 1.1. Vector-Vector Multiplication
# 
# Given two vectors $x,y \in \mathbb{R}^n$, the quantity $x^Ty$, sometimes called the <span class = 'high'>inner product or dot product</span> of the vectors, is a real number given by
# 
# <br>
# 
# $$\begin{align} x^Ty \in \mathbb{R} = \begin{matrix} [x_{1} & x_{2} & ... & x_{n}]\end{matrix} \begin{bmatrix}
#       y_{1} \\
#       y_{2} \\
#       .. \\
#       y_{n}
#       \end{bmatrix} = \sum_{i=1}^{n}x_{i}y_{i}\end{align}$$
# 
# <br>
# 
# Given vectors $x \in \mathbb{R}^m, y \in \mathbb{R}^n$ (not necessarily of the same size), $xy^T \in \mathbb{R}^{m \times n}$ is called the <span class = 'high'>outer product</span> of the vectors. It is a matrix, whose entries are given by $(xy^T)_{ij} = x_iy_j$:
# 
# <br>
# 
# $$xy^T \in \mathbb{R}^{m \times n} = \begin{bmatrix}
#       x_{1} \\
#       x_{2} \\
#       .. \\
#       x_{n}
#       \end{bmatrix}\begin{matrix} [y_{1} & y_{2} & ... & y_{n}]\end{matrix} = \begin{bmatrix}
#       x_{1}y_{1} & x_{1}y_{2} & ... & x_{1}y_{n} \\
#       x_{2}y_{1} & x_{2}y_{2} & ... & x_{2}y_{n} \\
#       ... & ... & ... & ... \\
#       x_{m}y_{1} & x_{m}y_{2} & ... & x_{m}y_{n} \\
#       \end{bmatrix}$$
#       
# <br>
#       
# ### 1.2. Matrix-Vector Products
# 
# Given a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $x \in \mathbb{R}^n$, their product is a vector $y = Ax \in \mathbb{R}^m$. There are a couple of ways of looking at matrix-vector multiplication, and we will look at each of them in turn.
# 
# If we write $A$ by rows, then we can express $Ax$ as,
# 
# <br>
# 
# $$y = Ax = \begin{bmatrix}
#       - a_{1}^T - \\
#       - a_{2}^T - \\
#       .. \\
#       - a_{m}^T - 
#       \end{bmatrix}x = \begin{bmatrix}
#       a_{1}^Tx\\
#       a_{2}^Tx \\
#       .. \\
#       a_{m}^Tx 
#       \end{bmatrix}$$
# 
# <br>
# 

# In other words, the $i$th entry of $y$ is equal to the inner product of the $i$th row of $A$ and $x$, $y_i=a_i^Tx$.
# 
# Alternatively, let's write $A$ in column form. In this case we see that,
# 
# <br>
# 
# $$ y = Ax = \begin{bmatrix}
#       | & | & | & ... & | \\
#       a^1 & a^2 & a^3 & ... & a^n \\
#       | & | & | & ... &|
#       \end{bmatrix} \begin{bmatrix}
#       x_{1} \\
#       x_{2} \\
#       x_{3} \\
#       .. \\
#       x_{n}
#       \end{bmatrix} = \begin{matrix}
#       a^1
#       \end{matrix}x_1+\begin{matrix}
#       a^2
#       \end{matrix}x_2+...+\begin{matrix}
#       a^n
#       \end{matrix}x_n \color{blue}{\enspace \rightarrow \enspace (1)}
# $$
# 
# $$ y = Ax = \begin{bmatrix}
#       | & | & | & ... & | \\
#       a^1 & a^2 & a^3 & ... & a^n \\
#       | & | & | & ... &|
#       \end{bmatrix} \begin{bmatrix}
#       x_{1} \\
#       x_{2} \\
#       x_{3} \\
#       .. \\
#       x_{n}
#       \end{bmatrix} = \begin{matrix}
#       a^1
#       \end{matrix}x_1+\begin{matrix}
#       a^2
#       \end{matrix}x_2+...+\begin{matrix}
#       a^n
#       \end{matrix}x_n
# $$ (1)
# 
# <br>
#       
# In other words, $y$ is a **linear combination** of the columns of $A$, where the coefficients of the linear combination are given by the entries of $x$.
# 
# ### 1.3. Matrix-Matrix Products
# 
# Using the above information, we can view matrix-matrix multiplication from various perspectives. One of them is to view the matrix-matrix multiplication as a set of vector-vector products. The most obvious viewpoint, which follows immediately from the definition, is that the $(i,j)$th entry of $C$ is equal to the inner product of the $i$th row of $A$ and the $j$th column of $B$. Symbolically, this looks like the following:
# 
# <br>
# 
# $$C = AB = \begin{bmatrix}
#       - a_{1}^T - \\
#       - a_{2}^T - \\
#       .. \\
#       - a_{m}^T - 
#       \end{bmatrix} \begin{bmatrix}
#       | & | & | & ... & | \\
#       b^1 & b^2 & b^3 & ... & b^n \\
#       | & | & | & ... &|
#       \end{bmatrix}=\begin{bmatrix}
#       a_1^Tb^1 & a_1^Tb^2 & ... & a_1^Tb^p \\
#       a_2^Tb^1 & a_2^Tb^2 & ... & a_2^Tb^p \\
#       .. & .. & ... & .. \\
#       a_m^Tb^1 & a_m^Tb^2 & ... & a_m^Tb^p
#       \end{bmatrix} $$
#       
# <br>
#       
# Remember that since $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}, a_i \in \mathbb{R}^n \text{ and } b^j \in \mathbb{R}^n$, so these inner products all make sense. This is the 'natural' representation when we represent $A$ by rows and $B$ by columns. Alternatively, we can represent $A$ by columns, and $B$ by rows. This representation leads to a much tricker interpretation of $AB$ as a sum of outer products. Symbolically,
# 
# <br>
# 
# $$ C = AB = \begin{bmatrix}
#       | & | & | & ... & | \\
#       a^1 & a^2 & a^3 & ... & a^n \\
#       | & | & | & ... &|
#       \end{bmatrix} \begin{bmatrix}
#       - b_{1}^T - \\
#       - b_{2}^T - \\
#       .. \\
#       - b_{n}^T - 
#       \end{bmatrix} = \sum_{i=1}^{n}a^ib_{i}^T $$
# 
# <br>
# 
# Put another way, $AB$ is equal to the sum, over all $i$, of the outer product of the $i$th column of $A$ and the $i$th row of $B$. Since, in this case, $a^i \in \mathbb{R}^m$ and $b_i \in \mathbb{R}^p$, the dimension of the outer product $a^ib_i^T$ is $m \times p$, which coincides with the dimension of $C$. Chances are, the last equality above may appear confusing to you. If so, take the time to check it for yourself!
# 
# Second, we can also view matrix-matrix multiplication as a set of matrix-vector products. Specifically, if we represent $B$ by columns, we can view the columns of $C$ as matrix-vector products between $A$ and the columns of $B$. Symbolically,
# 
# <br>
# 
# $$
# C = AB = A \begin{bmatrix}
#       | & | & | & ... & | \\
#       b^1 & b^2 & b^3 & ... & b^n \\
#       | & | & | & ... &|
#       \end{bmatrix} = \begin{bmatrix}
#       | & | & | & ... & | \\
#       Ab^1 & Ab^2 & Ab^3 & ... & Ab^n \\
#       | & | & | & ... &|
#       \end{bmatrix} \color{blue}{\enspace \rightarrow \enspace (2)}
# $$
# 
# <br>
# 

# ## 2. Operations and Properties
# 
# ### 2.1. Symmetric Matrices
# 
# A square matrix $A \in \mathbb{R}^{n \times n}$ is **symmetric** if $A=A^T$. It is **anti-symmetric** if $A=-A^T$. It is easy to show that any matrix $A \in \mathbb{R}^{n \times n}$, the matrix $A+A^T$ is symmetric and the matrix $A-A^T$ is anti-symmetric. From this it follows that any square matrix $A \in \mathbb{R}^{n \times n}$ can be be represented as a sum of a symmetric matrix and an anti-symmetric matrix as shown below:
# 
# <br>
# 
# $$A = \frac{1}{2}(A+A^T)+\frac{1}{2}(A-A^T)$$
# 
# <br>
# 

# ### 2.2. Norms
# 
# A norm, $||x||$, of a vector is informally defined as the measure of 'length' of the vector. For example, we have the commonly used Eucledian or $l_{2}$ norm,
# 
# <br>
# 
# $$||x||_{2} = \sqrt{\sum_{i=1}^{n}x_{i}^2}$$
# 
# <br>
# 
# Note that $||x||_{2}^2 = x^Tx$
# 
# More formally, norm is a function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ that satisfies $4$ properties:
# 
# <p style="line-height:180%;">
#     
# $\rightarrow \text{For all } x \in \mathbb{R}^{n}, f(x) \geq 0. \text{ (non-negativity)}$
#     
# $\rightarrow f(x)=0 \text{ if and only if } x=0 \text{ (definiteness)}$
# 
# $\rightarrow \text{For all } x \in \mathbb{R}^{n}, t \in \mathbb{R}, f(tx) = |t|f(x). \text{ (homogeneity)}$
# 
# $\rightarrow \text{For all } x,y \in \mathbb{R}^{n}, f(x+y) \leq f(x) +f(y). \text{ (traingle inequality)}$
# </p>
# 
# Other examples of norms are the $l_1$ norm,
# 
# $$||x||_1 = \sum_{i=1}^{n}|x_i|$$
# 
# and the $l_\infty$ norm,
# 
# $$||x||_\infty = max_i|x_i|$$
# 
# In fact, all three norms presented so far are examples of the family of $l_p$ norms, which are parameterized by a real number $p \geq 1$ and defined as
# 
# $$||x||_p = \left( \sum_{i=1}^n|x_i|^p\right)^{1/p}$$
# 
# <br>

# Norms can also be defined for matrices, such as the Frobenius norm,
# 
# $$\begin{align}||A||_{F} = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}A_{ij}^2} = \sqrt{trace(A^TA)}\end{align}$$
# 
# <br>
# 
# ### 2.3. Linear Independence and Rank
# 
# A set of vectors $\{x_1, x_2,..,x_n\} \subset \mathbb{R}^n$ is set to be **linearly independent** if no vector can be represented as a linear combination of the remaining vectors. Conversely, if one vector belonging to the set can be represented as a linear combination of the remaining vectors, then the vectors are said to be **linearly dependent**. That is, if
# 
# $$x_n = \sum_{i=1}^{n-1}\alpha_i x_i$$
# 
# <br>
# 
# for some scalar values $\alpha_1,...,\alpha_{n-1} \in \mathbb{R}$,then we say that the vectors $\{x_1, x_2,..,x_n\}$ are linearly dependent; otherwise, the vectors are linearly independent. For example, the vectors
# 
# <br>
# 
# $$x_1=\begin{bmatrix}
# 1 \\
# 2\\
# 3\end{bmatrix}\enspace x_2 = \begin{bmatrix}
# 4 \\
# 1\\ 
# 5 \end{bmatrix}\enspace x_3 = \begin{bmatrix}
# 2 \\
# -3 \\
# -1\end{bmatrix}$$
# 
# <br>
# 
# are linearly dependent because $x_3 = -2x_1+x_2$.
# 
# **_Column Rank_**
# 
# The column rank of a matrix $A \in \mathbb{R}^{m \times n}$ is the size of the largest subset of columns of $A$ that constitute a linearly independent set.
# 
# **_Row Rank_**
# 
# The row rank of a matrix $A \in \mathbb{R}^{m \times n}$ is the size of the largest subset of rows of $A$ that constitute a linearly independent set.
# 
# If, for matrix column rank is equal to row rank, then both quantities are collectively referred to as the **rank of the matrix $A$**
# 
# <p style="line-height:180%;">
#     
# $\rightarrow \text{For } A \in \mathbb{R}^{m \times n}, \text{ rank}(A) \leq min(m,n). \text{ If rank}(A) = min(m,n), \text{then } A \text{ is said to be } \textbf{full rank}$
# 
# $\rightarrow \text{For } A \in \mathbb{R}^{m \times n}, \text{ rank}(A) = \text{ rank}(A^T)$
# 
# $\rightarrow \text{For } A,B \in \mathbb{R}^{m \times n},\text{ rank}(A+B) \leq \text{ rank}(A) + \text{ rank}(B)$
# </p>
# 
# <br>
# 

# ### 2.4. The Inverse of a Square Matrix
# 
# The **inverse** of a square matrix $A \in \mathbb{R}^{n \times n}$ is denoted $A^{-1}$, and is the unique matrix such that:
# 
# $$A^{-1}A=I=AA^{-1}$$
# 
# Note that not all matrices have inverses. Non-square matrices, for example, do not have inverse by definition. However, for some square matrices $A$, it may stil be the case that $A^{-1}$ may not exist. In particular, we say that $A$ is **invertible** or **non-singular** if $A^{-1}$ exists and **non-invertible** or **singular** otherwise.
# 
# In order for a square matrix $A$ to have an inverse $A^{-1}$, $A$ must be of full rank.
# 
# The following are properties of the inverse; all assume that $A, B \in \mathbb{R}^{n \times n}$ are non-singular:
# 
# <p style="line-height:180%;">
#     
# $\rightarrow (A^{-1})^{-1}=A$
# 
# $\rightarrow (AB)^{-1} = B^{-1}A^{-1}$
# 
# $\rightarrow (A^{-1})^T = (A^T)^{-1}$
# 
# $(A^T)^{-1}$ is denoted by $A^{-T}$
# </p>
# 
# <br>
# 
# ### 2.5. Orthogonal Matrices
# 
# Two vectors $x,y \in \mathbb{R}^n$ are **orthogonal** if $x^Ty=0$. A vector $x \in \mathbb{R}^n$ is **normalized** if $||x||_2=1$. A square matrix $U \in \mathbb{R}^{n \times n}$ is **orthogonal** _(note the different meanings when talking about vectors versus matrices)_ if all its columns are orthogonal to each other and are normalized.
# 
# It follows immediately from the definition of orthogonality and normality that
# 
# <br>
# 
# $$U^TU = I = UU^T$$
# 
# <br>
# 
# In other words, the inverse of an orthogonal matrix is its transpose. Note that if $U$ is not square (i.e., $U \in \mathbb{R}^{m \times n}, \enspace n < m$) but its columns are still orthonormal, then $U^TU=I$, but $UU^T \neq I$.
# 
# Another nice property of orthogonal matrices is that operating on a vector with an orthogonal matrix will not change its _Euclidean norm_. i.e.,
# 
# $$||Ux||_2 = ||x||_2 \color{blue}{\enspace \rightarrow \enspace (3)}$$
# 
# <br>
# 
# for any $x \in \mathbb{R}^n, U \in \mathbb{R}^{n \times n}$ orthogonal.
# 
# ### 2.6. Range and nullspace of a Matrix
# 
# The span of a set of vectors $\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix}$ is the set of all vectors that can be expressed as a linear combination of $\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix}$. That is,
# 
# <br>
# 
# $$\text{span}(\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix}) = \left \{ v:v = \sum_{i=1}^{n} \alpha_ix_i, \enspace \alpha_i \in \mathbb{R} \right \}$$
# 
# <br>
# 
# It can be shown that if $\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix}$ is a set of $n$ linearly independent vectors, where each $x_i \in \mathbb{R}^n$, then $\text{span}(\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix}) = \mathbb{R}^n$. In other words, any vector $v \in \mathbb{R}^n$ can be written as a linear combination of $x_1$ through $x_n$.
# 
# The **projection** of a vector $y \in \mathbb{R}^m$ onto the span of $\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix}$ (here we assume $x_i \in \mathbb{R}^m$) is a vector $v \in \text{span}(\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix})$, such that $v$ is as close as possible to $y$, as measured by the Euclidean norm $||v-y||_2$. We denote the projection as $\text{Proj}(y;\begin{Bmatrix} x_{1},x_{2},...x_{n} \end{Bmatrix})$ and can define it formally as,
# 
# <br>
# 
# $$\text{Proj}(y;\begin{Bmatrix}x_{1},...,x_{n}\end{Bmatrix}) = \text{argmin}_{v \in span(\begin{Bmatrix}x_{1},...,x_{n}\end{Bmatrix})}||y-v||_{2}$$
# 
# <br>
# 
# The range of a matrix $A \in \mathbb{R}^{m \times n}$, denoted by $R(A)$, is the span of the columns of $A$. In other words,
# 
# <br>
# 
# $$R(A) = \begin{Bmatrix}v \in \mathbb{R}^m : v = Ax, x \in \mathbb{R}^n \end{Bmatrix}$$
# 
# <br>
# 
# The nullspace of a matrix $A \in \mathbb{R}^{m \times n}$, denoted by $\mathcal{N}(A)$ is the set of all vectors that equal to $0$ when multiplied by $A$, i.e.,
# 
# $$\mathcal{N}(A) = \begin{Bmatrix} x \in \mathbb{R}^n : Ax = 0\end{Bmatrix}$$
# 
# <br>
# 

# ### 2.7. Quadratic Forms and Positive Semidefinite Matrices
# 
# Given a square matrix $A \in \mathbb{R}^{n \times n}$ and a vector $x \in \mathbb{R}^{n}$, the scalar value $x^TAx$ is called a _quadratic form_. Written explicitly, we see that
# 
# <br>
# 
# $$\begin{align}x^TAx = \sum_{i=1}^{n}x_{i}(Ax)_{i} = \sum_{i=1}^{n}x_{i}\left (\sum_{j=1}^{n}A_{ij}x_{j}\right ) = \sum_{i=1}^{n}\sum_{j=1}^{n}A_{ij}x_{i}x_{j}\end{align}$$
# 
# <br>
# 
# Note that,
# 
# $$x^TAx = (x^TAx)^T = x^TA^Tx = x^T \left( \frac{1}{2}A + \frac{1}{2}A^T\right)x$$
# 
# <br>
# 
# where the first equality follows from the fact that the transpose of a scalar is equal to itself, and the second equality follows from the fact that we are averaging two quantities which are themselves equal. From this, we can conclude that only the symmetric part of $A$ contributes to the quadratic form. For this reason, we often implicitly assume that the matrices appearing in the quadratic form are symmetric.
# 
# We have the following definitions:
# 
# * A symmetric matrix $A \in \mathbb{S}^n$ is <span class = 'high'>positive definite</span> (PD) if for all non-zero vectors $x \in \mathbb{R}^n, x^TAx>0$. This is usually denoted $A\succ0$, and often times the set of all positive definite matrices is denoted by $\mathbb{S}_{++}^n$
# 
# <br>
# 
# * A symmetric matrix $A \in \mathbb{S}^n$ is <span class = 'high'>positive semidefinite</span> (PSD) if for all non-zero vectors $x \in \mathbb{R}^n, x^TAx \geq 0$. This is usually denoted $A \succeq 0$, and often times the set of all positive semidefinite matrices is denoted by $\mathbb{S}_{+}^n$
# 
# <br>
# 
# * A symmetric matrix $A \in \mathbb{S}^n$ is <span class = 'high'>negative definite</span> (ND), denoted $A\prec0$ if for all non-zero vectors $x \in \mathbb{R}^n, x^TAx<0$.
# 
# <br>
# 
# * A symmetric matrix $A \in \mathbb{S}^n$ is <span class = 'high'>negative semidefinite</span> (NSD), denoted $A \preceq 0$ if for all non-zero vectors $x \in \mathbb{R}^n, x^TAx \leq 0$.
# 
# <br>
# 
# * A symmetric matrix $A \in \mathbb{S}^n$ is <span class = 'high'>indefinite</span>, if it is neither positive semidefinite nor negative semidefinite - i.e., if there exists $x_{1}, x_{2} \in \mathbb{R}^n $ such that $x_{1}^TAx_{1}>0$ and $x_2^TAx_2<0$.
# 
# It should be obvious that if $A$ is positive definite, then $-A$ is negative definite and vice versa. Likewise, if $A$ is positive semidefinite then $-A$ is negative semidefinite and vice versa. If $A$ is indefinite, then so is $-A$.
# 
# One important property of positive definite and negative definite matrices is that they are always full rank, and hence, invertible. To see why this is the case, suppose that some matrix $A \in \mathbb{R}^{n \times n}$ is not full rank. Then, suppose that the $j$th column of $A$ is expressible as a linear combination of other $n-1$ columns:
# 
# $$a_j = \sum_{i \neq j} x_ia_i$$
# 
# for some $x_1,..x_{j-1},x_{j+1},...,x_n \in \mathbb{R}$. Setting $x_j=-1$, we have
# 
# $$Ax = \sum_{i=1}^n x_ia_i=0$$
# 
# But this implies $x^TAx=0$ for some non-zero vector $x$, so $A$ must be neither positive definite nor negative definite. Therefore, if $A$ is either positive definite or negative definite, it must be full rank.
# 
# Finally, there is one type of positive definite matrix that comes up frequently, and so deserves some special mention. Given any matrix $A \in \mathbb{R}^{m \times n}$ (not necessarily symmetric or even square), the matrix $G = A^TA$ (sometimes called a **Gram Matrix**) is always positive semidefinite. Further, if $m \geq n$ (and we assume for convenience that A is full rank), then $G=A^TA$ is positive definite.

# <br>
# 
# ### 2.8. Eigenvalues and Eigenvectors
# 
# Given a square matrix $A \in \mathbb{R}^{n \times n}$, we say that $\lambda \in \mathbb{C}$ is an **eigenvalue** if $A$ and $x \in \mathbb{C}^n$ is the corresponding **eigenvector** if 

# $$ Ax = \lambda x, \enspace x \neq 0 $$
# 
# Intuitively, this definition means that multiplying $A$ by the vector $x$ results in a new vector that points in the same direction as $x$, but scaled by a factor $\lambda$. Also note that for any eigenvector $x \in \mathbb{C}^n$ and scalar $t \in \mathbb{C}, A(cx) = cAx = c \lambda x = \lambda (cx)$, so $cx$ is also an eigenvector. For this reason when we talk about 'the' eigenvector associated with $\lambda$, we usually assume that the eigenvector is normalized to have length $1$.
# 
# We can rewrite the equation above to state that $(\lambda, x)$ is an eigenvalue-eigenvector pair of $A$ if,
# 
# $$(\lambda I -A)x =0, \enspace x \neq 0$$
# 
# <br>
# 
# But $(\lambda I -A)x =0$ has a non-zero solution to $x$ if and only if $(\lambda I -A)$ has a non-empty nullspace, which is only the case if $(\lambda I -A)$ is singular, i.e.,
# 
# $$|(\lambda I -A)|=0$$
# 
# <br>

# We can now use the definition of the determinant to expand this expression $|(\lambda I -A)|$ into a polynomial in $\lambda$, where $\lambda$ will have degree $n$. It's often called the characteristic polynomial of the matrix $A$.
# 
# We then find the $n$ roots of this characteristic polynomial and denote them by $\lambda_1, \lambda_2,...,\lambda_n$. These are all the eigenvalues of the matrix $A$.
# 
# The following are the properties of eigenvalues and eigenvectors:
# 
# * The trace of $A$ is equal to the sum of its eigenvalues,
# 
# $$trace(A) = \sum_{i=1}^n \lambda_i$$
# 
# * The determinant of $A$ is equal to the product of its eigenvalues,
# 
# $$|A| = \prod_{i=1}^n \lambda_i$$
# 
# * The rank of $A$ is equal to the number of non-zero eigenvalues of $A$.
# 
# <br>
# 
# * Supppose $A$ is non-singular with eigenvalue $\lambda$ and an associated eigenvector $x$. Then $\frac{1}{\lambda}$ is an eigenvalue of $A^{-1}$ with an associated eigenvector $x$, i.e., $A^{-1}x = (\frac{1}{\lambda})x$.
# 
# <br>
# 
# * The eigenvalues of a diagonal matrix $D = \text{diag}(d_1,...,d_n)$ are just the diagonal entries $d_1,...,d_n$

# <br>
# 
# ### 2.9. Eigenvalues and Eigenvectors of Symmetric Matrices
# 
# In general, the structures of the eigenvalues and eigenvectors of a general square matrix can be subtle to characterize. Fortunately, in most of the cases in machine learning, if suffices to deal with symmetric real matrices, whose eigenvalues and eigenvectors have remarkable properties.
# 
# Throughout this section, let's assume that $A$ is a symmetric real matrix. We have the following properties:
# 
# * All eigenvalues of $A$ are real numbers. We denote them by $\lambda_1,...,\lambda_n$
# 
# <br>
# 
# * There exists a set of eigenvectors $u_1,...,u_n$ such that:
# 
# a) For all $i, u_i$ is an eigenvector with eigenvalue $\lambda_i$ and
# 
# b) $u_1,...,u_n$ are unit vectors and orthogonal to each other.
# 
# Let $U$ be the orthonormal matrix that contains $u_i$'s as columns:
# 
# <br>
# 
# $$U = \begin{bmatrix}
#       | & | & | & ... & | \\
#       u^1 & u^2 & u^3 & ... & u^n \\
#       | & | & | & ... &|
#       \end{bmatrix} \color{blue}{\enspace \rightarrow \enspace (4)}$$
#       
# <br>
#       
# Let $\Lambda = \text{diag}(\lambda_1,...,\lambda_n)$ be the diagonal matrix that contains $\lambda_1,...,\lambda_n$ as entries on the diagonal. Using the view of matrix-matrix vector multiplication in <a href="#matrix-matrix-products" >$\small{\color{blue}{\text{equation }(2)}}$</a>, we can verify that
#       
# <br>
# 
# $$ \begin{align}AU = \begin{bmatrix}
#       | & | & | & ... & | \\
#       Au^1 & Au^2 & Au^3 & ... & Au^n \\
#       | & | & | & ... &|
#       \end{bmatrix} &= \begin{bmatrix}
#       | & | & | & ... & | \\
#       \lambda_1u^1 & \lambda_2u^2 & \lambda_3u^3 & ... & \lambda_nu^n \\
#       | & | & | & ... &|
#       \end{bmatrix} \\ \\ &= U\text{diag}(\lambda_1,...,\lambda_n) \\ \\&= U\Lambda \end{align}$$
#             
# <br>
#       
# Recalling that orthonormal matrix $U$ satisfies $UU^T=I$ and using the equation above, we have
#       
# <br>
# 
# $$A = AUU^T = U \Lambda U^T \color{blue}{\enspace \rightarrow \enspace (5)}$$
#       
# <br>
# 
# This new presentation of $A$ as  $U \Lambda U^T$ is often called <span class = 'high'>diagonalization</span> of the matrix $A$. The term diagonalization comes from the fact that with such representation, we can often effectively treat a symmetric matrix $A$ as a diagonal matrix - which is much easier to understand - w.r.t. the basis defined by the eigenvectors $U$. We will elaborate this below by several examples.
# 
# _Background: representing vector w.r.t. another basis:_
#       
# <br>
# 
# Any orthonormal matrix $U = \begin{bmatrix}
#       | & | & | & ... & | \\
#       u^1 & u^2 & u^3 & ... & u^n \\
#       | & | & | & ... &|
#       \end{bmatrix}$ defines a new basis (coordinate system) of $\mathbb{R}^n$ in the following sense. For any vector $x \in \mathbb{R}^n$ can be represented as a linear combination of $u_1,...,u_n$ with coefficient $\hat{x_1},...,\hat{x_n}$
# 
# $$x = \hat{x_1}u_1+...+\hat{x_n}u_n = U\hat x$$
# 
# where in the second equality we use the view of <a href="#matrix-vector-products" >$\small{\color{blue}{\text{equation }(1)}}$</a>. Indeed, such $\hat{x}$ uniquely exists
# 
# $$x = U\hat{x} = U^Tx = \hat{x}$$

# In other words, the vector $\hat x = U^Tx$ can serve as another representation of the vector $x$ w.r.t. the basis defined by $U$.
#       
# <br>
# 
# ### 2.10. Diagonalizing matrix-vector multiplication:
# 
# With the setup above, we will see that left-multiplying matrix $A$ can be viewed as left-multiplying a diagonal matrix w.r.t. the basis of eigenvectors. Suppose $x$ is a vector and $\hat x$ is its representation w.r.t. to the basis of $U$. Let $z = Ax$ be the matrix-vector product. Now, let's compute the representation $z$ w.r.t. the basis of $U$:
# 
# Then, again using the fact that $UU^T=U^TU=I$ and <a href="#eigenvalues-and-eigenvectors-of-symmetric-matrices" >$\small{\color{blue}{\text{equation }(5)}}$</a>, we have that
#       
# <br>
# 
# $$\hat z = U^Tz = U^TAx = U^TU\Lambda U^Tx = \Lambda\hat x = \begin{bmatrix}
# \lambda_1 \hat{x_1} \\
# \lambda_2 \hat{x_2}\\
# ... \\
# \lambda_n \hat{x_n}\end{bmatrix}$$
#       
# <br>
# 
# We see that left-multiplying matrix $A$ in the original space is equivalent to left-multiplying the diagonal matrix $\Lambda$ w.r.t the new basis, which is merely scaling each coordinate by the value of corresponding eigenvalue.
# 
# Under the new basis, multiplying a matrix multiple times becomes much simpler as well. For example, suppose $q = AAAx$. Deriving out the analytical form of $q$ in terms of the entries of $A$ may be a nightmare under the original basis, but can be much easier under the new one:
#       
# <br>
# 
# $$\hat{q} = U^Tq = U^TAx = U^TU\Lambda U^TU\Lambda U^TU\Lambda U^Tx = \Lambda^3\hat x = \begin{bmatrix}
# \lambda_1^3 \hat{x_1} \\
# \lambda_2^3 \hat{x_2}\\
# ... \\
# \lambda_n^3 \hat{x_n}\end{bmatrix} \color{blue}{\enspace \rightarrow \enspace (6)}$$
#       
# <br>
# 
# **"Diagonalizing" quadratic form:**
# 
# As a directly corollary, the quadratic form $x^TAx$ can also be simplified under the new basis.
#       
# <br>
# 
# $$x^TAx = x^TU\Lambda U^Tx = \hat x \Lambda \hat x = \sum_{i=1}^{n}\lambda_i \hat{x_i}^2 \color{blue}{\enspace \rightarrow \enspace (7)}$$
#       
# <br>
# 
# (Recall that with the old representation, $x^TAx = \sum_{i=1, j=1}^n x_i x_j A_{ij}$ involves a sum of $n^2$ terms instead of $n$ terms in the equation above.) With this viewpoint, we can also show that the definiteness of the matrix $A$ depends entirely on the sign of its eigenvalues:
# 
# <p style="line-height:180%;">
#     
# $\rightarrow \text{If all } \lambda_i > 0 \text{ then the matrix } A \text{ is positive definite because:}$
# 
# $$x^TAx = \sum_{i=1}^n \lambda_i \hat{x_i}^2 > 0 \text{ for any } \hat x \neq 0$$
#     
# $\rightarrow \text{If all } \lambda_i \geq 0 \text{ then the matrix } A \text{ is positive semi-definite because:}$
#     
# $$X^TAx  = \sum_{i=1}^n\lambda_i \hat{x_i}^2 \geq 0 \text{ for all }\hat x$$
#     
# $\rightarrow \text{Likewise, if all } \lambda_i <0 \text{ or } \lambda_i \leq 0 \text{, then A is negative definite or negative semidefinite respectively.}$
# 
# $\rightarrow \text{Finally, if A has both positive and negative eigenvalues, say } \lambda_i >0 \text{ and } \lambda_j <0,$
#     
# $\text{then it is indefinite.}$
#     
# This is because if we let $\hat x$ satisfy $\hat x_i = 1$ and $\hat x_k = 0, \enspace \forall k\neq i$, then $x^TAx = \sum_{i=1}^{n}\lambda_i \hat{x_i}^2>0$. Similarly, we can let $\hat x$ satisfy $\hat{x_j}=1$ and $\hat{x_k}=0, \enspace \forall k \neq j$, then $x^TAx = \sum_{i=1}^{n} \lambda_i \hat{x_i}^2 <0$.
# 
# </p>
# 
# An application where eigenvalues and eigenvectors come up frequently is in maximizing some function of a matrix. In particular, for a matrix $A \in \mathbb{S}^n$, consider the following maximization problem,
#       
# <br>
# 
# $$\text{max}_{x \in \mathbb{R}^n} \enspace x^TAx = \sum_{i=1}^{n} \lambda_i \hat{x_i}^2, \enspace \text{subject to } ||x||_2 ^ 2=1 \color{blue}{\enspace \rightarrow \enspace (8)}$$
#       
# <br>
# 
# i.e., we want to find the vector which maximizes the quadratic form. Assuming the eigenvalues are ordered as $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n$, the optimal values of this optimization problem is $\lambda_1$ and any eigenvector $u_1$ corresponding to $\lambda_1$ is one of the maximizers.
# 
# We show this by using the diagonalization technique: Note that $||x||_2=||\hat x||_2$ by <a href="#orthogonal-matrices" >$\small{\color{blue}{\text{equation }(3)}}$</a>, and using <a href="#diagonalizing-matrix-vector-multiplication" >$\small{\color{blue}{\text{equation }(7)}}$</a>, we can rewrite the optimization <a href="#diagonalizing-matrix-vector-multiplication" >$\small{\color{blue}{\text{equation }(8)}}$</a> as:
#       
# <br>
# 
# $$\text{max}_{x \in \mathbb{R}^n} \enspace \hat{x}^T\Lambda \hat{x} = \sum_{i=1}^n \lambda_i\hat{x_i}^2 \enspace \text{ subject to } ||\hat x||_2^2=1 \color{blue}{\enspace \rightarrow \enspace (9)}$$
#       
# <br>
# 
# Then, we have that the objective is upper bounded by $\lambda_1$:
#       
# <br>
# 
# $$\hat{x}^T \Lambda \hat x = \sum_{i=1}^n \lambda_i \hat{x_i}^2 \leq \sum_{i=1}^n \lambda_1 \hat{x_i}^2 = \lambda_1 \color{blue}{\enspace \rightarrow \enspace (10)}$$
#       
# <br>
# 
# Moreover, setting $\hat{x}=\begin{bmatrix}
# 1 \\
# 0\\
# ..\\
# 0\end{bmatrix}$ achieves the equality in the equation above, and this corresponds to setting $x=u_1$

# 

# In[ ]:





# In[ ]:





# In[ ]:




