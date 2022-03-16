#!/usr/bin/env python
# coding: utf-8

# #### Lecture 1: Introduction to Machine Learning

# The term 'Machine Learning' was coined by <span class = 'high'>Arthur Samuel</span>. He wrote a program that would learn to play [checkers](https://en.wikipedia.org/wiki/Checkers) by playing against itself without any kind of strategy being explicitly programmed into it. The program learned to improve itself by gaining experience by playing more and more games. This program by Arthuer Samuel is considered to be the start of Machine Learning.

# A more common definition by <span class = 'high'>Tom Mitchell</span> (a professor at CMU) defines Machine Learning as:
# 
# ```{admonition} Definition
# :class: note
# A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
# ```
# 
# _One might say 'experience' is a vague term. Experience in this context means prior data. The algorithm analyzes this prior data  (training data), looks for patterns in the data and the algorithm improves it's performance at the task it was designed to operate on._
# 
# ```{admonition} Tip
# :class: tip
# The key here is the data (the experience from the past) that distinguishes machine learning from other fields.
# ```
# 
# Machine Learning is also related to Artificial Intelligence (AI). AI is a broader field. Machine Learning is a subset of AI. AI is a field about building programs that perform at the level of human cognition. Machine Learning is one approach of how to implement these programs.
# 
# Within Machine Learning, a field called Deep Learning has seen major advancements in recent times.

# **Examples of fields where Machine Learning has made a lot of progress:**
# 
# * _Computer Vision and Image Recognition_
# 
# The ImageNET dataset and the Convolutional Neural Networks more or less revolutionized Computer Vision. This field has also improved autonomous driving where machine learning techniques are used to sense to 'stop' signs, where the pedestrians are and so on.
# 
# * _Speech Recognition_
# 
# Machine Learning made things like Voice Assistants, Language Translation (Deep Learning), Unsupervised Translation.
# 
# * _Reinforcement Learning and Deep Reinforcement Learning_
# 
# The majority of the progress has been in gameplaying. For example DeepMind built RL models that could play atari games, AlphaGO.
# 
# _This course starts with the basics of Machine Learning, Deep Learning, and touches upon the fundamentals of Reinforcement Learning._
# 
# **Supervised Machine Learning**
# 
# Supervised machine learning is a technique where the training data comes in pair. Each pair has an input and an output. The input is what you feed into the machine learning algorithm and the output is the desired result that the ML algorithm should yield. For each example, the ML model is told what the desired output must be. That's why its called Supervised ML.
# 
# **Unsupervised Machine Learning**
# 
# In this type of ML, the model is given some data and there is no supervision. You just ask the model to look for patterns in the data. The output of your algorithm is the interesting structure or pattern that the model derives from the data. These algorithms are further divided into the ones that look for clusters vs the ones that look for subspaces.
# 
# **Deep Learning**
# 
# Deep Learning (also called as Representation Learning) is the technique where you want the model to learn the representation of the data, while in non-deep learning approaches, we feed the representation of the data to the model. In Deep Learning we let the algorithm itself learn the representation of the data.
# 
# **Learning Theory**
# 
# This section deals with the super important concepts of `bias-variance trade-off`, `generalization` and `uniform convergence`. How do we expect a model that is trained on a particular dataset to even perform at any level of accuracy when used on production. This is the part that makes this course very interesting because these principles are common accross all the learning algorithms and the alogorithms that have not been invented yet. These are the foundational aspects of why Machine Learning works.
# 
# _Finally, we touch upon some fundamentals of Reinforcement Learning_.
# 
# **Examples**
# 
# * Image classification
# 
# This is an example of Supervised ML algorithm where the model fed images (in terms of pixel values) and the model is expeccted to recognise the class of that input. One example dataset to image classification is the classic <span class = 'high'>MNIST</span> digit recoginition dataset.

# ```{figure} ../images/mnist.jpg
# ---
# height: 350px
# width: 600px
# name: directive-fig
# ---
# MNIST dataset. [credits](https://storage.googleapis.com/kaggle-datasets-images/1488071/2458428/b59f5c3418ee82f4271b402fb5cbad14/dataset-cover.jpg?t=2021-07-24-15-25-27"/)
# ```

# This is an example of multi-class classification problem where the input to the model is $28 \times 28$ image (pixel values ranging from 0 to 256) and the expected output is 'one of the numbers from 0 to 9'.
# 
# <blockquote>We shall be working on this dataset and implementing a neural network to build a hand-written digit recognition classifier.</blockquote>
# 
# * The cocktail problem (Unsupervised Machine Learning)
# 
# There are n speakers and microphones placed at specific locations. For simplicity let's assume $n=2$. So, there are two speakers and two microphones located at specific locations. Both these microphones capture overlapping sounds from both speakers. Our goal is to seperate these voices and produce their seperate waveforms.
# 
# We just provide two audio clips and there is no supervision whatsoever.

# * Cart-Pole Balancing problem (Reinforcement Learning)
# 
# ```{figure} https://www.researchgate.net/profile/Xiaolong-Ma-11/publication/6421220/figure/fig2/AS:601218367909895@1520353037352/Cart-pole-balancing-problem-The-force-applied-to-the-cart-is-atF-where-01-at-1-is.png
# ---
# height: 350px
# width: 600px
# name: directive-fig
# ---
# Cart-pole balancing problem. [credits](https://www.researchgate.net/profile/Xiaolong-Ma-11/publication/6421220/figure/fig2/AS:601218367909895@1520353037352/Cart-pole-balancing-problem-The-force-applied-to-the-cart-is-atF-where-01-at-1-is.png)
# ```
# We are aiming to control the cartpole that is placed on the stick by using direction and angle as parameters. For more information about Reinforcement Learning refer my lecture notes [here](https:/www.theessentialai.com/rl)
# 
# Each trial is referred as episode and the agent tries to learn better and better over multiple episodes.

# In[ ]:





# In[ ]:





# In[ ]:




