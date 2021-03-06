#!/usr/bin/env python
# coding: utf-8

# # Can 4 output neurons classify 10 classes?

# The first chapter of the <a href = "http://neuralnetworksanddeeplearning.com/chap1.html" target="_blank">"Neural Networks and Deep Learning"</a> book by Michael Nielsen teaches how to use neural networks to classify images in the MNIST dataset. While reading this chapter, a paragraph caught my eye.
# 
# :::{admonition} <a href = "http://neuralnetworksanddeeplearning.com/chap1.html" target="_blank">Source</a>
# :class: tip
#  "You might wonder why we use $10$ output neurons. After all, the goal of the network
# is to tell us which digit $(0,1,2,. . .,9)$ corresponds to the input image. A seemingly natural
# way of doing that is <span style= "color:#E83E8C;">to use just $4$ output neurons, treating each neuron as taking on a binary
# value, depending on whether the neuron’s output is closer to $0$ or to $1$.</span> Four neurons are
# enough to encode the answer, since $2^4 = 16$ is more than the $10$ possible values for the input
# digit. Why should our network use $10$ neurons instead? Isn’t that inefficient?"
# :::
# 
# In this article we shall implement a neural network with just $4$ neurons to classify $10$ classes, using the above idea. <span style= "color:#E83E8C;">Can sigmoid perform multi-class classification?</span> Hold that thought!
# 
# <hr>
# 
# **This article:**
# <ol style = "line-height:180%;">
#     <li>Implements 4-neuron image classification model for MNIST dataset.</li>
# <li>Provides reasons for difference in performance of above model and the baseline 10 neuron model.</li>
#     <li>Extends the implementation of 4-neuron model to 16 classes.</li>
#     <li>Explains reasons, possible heuristics for difference in performances.</li>
#     </ol>
#     
# <hr>
# 
# ## Introduction
# 
# Our goal here is to perform Image Classification on the MNIST dataset using a neural network. Before proceeding to the code, this section discusses how neural networks learn.
# 
# ### Perceptron:
# 
# A perceptron takes in several binary inputs, $x_1, x_2, x_3,...,$ and produces a single binary output:
# 
# :::{figure-md} markdown-fig
# <img src="images/perceptron.png">
# 
#  Perceptron
# :::
# 
# 
# A simple way is introduced to find the output of perceptron. We assign weights,$w_1, w_2, w_3,...,$ to each input expressing the importance of respective inputs to the output. The perceptron's output, $0$ or $1$, is determined by whether the weighted sum $\sum_j w_jx_j$ is less than or greater than some <span style= "color:#E83E8C;">threshold</span> value. This equation can be re-written by introducing a bias term $b$ and classifying if the total sum is greater than $0$ or less than $0$. This is explained below in algebraic terms:
# 
# $$\begin{eqnarray}
#   \mbox{output} & = & \left\{ \begin{array}{ll}
#       0 & \mbox{if } \sum_j w_j x_j +b \leq 0 \\
#       1 & \mbox{if } \sum_j w_j x_j +b > 0
#       \end{array} \right.
# \end{eqnarray}$$
# 
# The perceptron can be thought as a unit that makes decisions by weighing up evidence.
# 
# ### Sigmoid Neuron:
# 
# The above form of perceptron has multiple caveats. We do not have control over the bias term $b$ as it can take any value on the number line as the numeric size of inputs changes. Additionally, a small change in the inputs can cause the perceptron to completely flip the class from $0$ to $1$. This leads to poor performance of model while training. To overcome this problem, the sum from the output is fed to a <span style= "color:#E83E8C;">sigmoid function.</span> This function is also called <span style= "color:#E83E8C;">an activation function</span>. Sigmoid functions takes any number and gives an output between $0$ and $1$. Hence $0.5$ can be used as a threshold to classify classes based on the output from sigmoid function.
# 
# $$\sigma(z) = \frac{1}{1+e^{-z}}$$
# 
# $$\text{output from sigmoid} = \frac{1}{1+\text{exp}(-\sum_jw_jx_j - b)}$$
# 
# <p style="text-align:center"><img src="images/sigmoid_step.PNG" height = "250px" width = "550px"/></p>
# 
# The above image compares the sigmoid function with the Step function. Sigmoid is a smoothed version of step function. Step-function can be considered to represent the perceptron except for $x=0$.
# 
# ### Learning process:
# 
# The smoothness introduced by the sigmoid function has a crucial advantage. The smoothness of $\sigma$ means that small changes $\Delta w_j$ in the weights and $\Delta b$ in the bias will produce a small change $\Delta \text{output}$ in the output from the neuron. The algebraic form is given by:
# 
# $$\Delta \text{output} \approx \sum_j\frac{\partial \text{ output}}{\partial w_j}\Delta w_j+\frac{\partial\text{ output}}{\partial b}\Delta b$$
# 
# For example, suppose the network was mistaken classifying an image as an “8” when it should be a “9”. We could figure out how to make a small change in the weights and biases so the network gets a little closer to classifying the image as a “9”. And then we’d repeat this, changing the weights and biases over and over to produce better and better output. The network would be learning.
# 
# <span style= "color:#E83E8C;">This is an high-level overview of the learning process.</span> More details about the learning process and backpropagation algorithm are saved for a future article.
# 
# ### Softmax neurons:
# 
# Softmax activation function comes into picture usually in the case of multi-class classficiation. Softmax outputs the probability of each class given an (image) input. The important difference between sigmoid and softmax function is that while sigmoid function outputs <span style= "color:#E83E8C;">confidence index of one class</span> in binary classification, softmax provides the probability of each class.
# 
# In the case of sigmoid it is interpreted that the likelihood of a class occuring in image is proportional to the value of sigmoid function but not exactly equal to the output. This leads to the fact that softmax activation function uses categorical cross-entropy loss with equal number of neurons as the number of classes.
# 
# ## Encoding image labels:
# 
# <span style= "color:#E83E8C; font-weight:bold;">This section is the core component of this article.</span>
# 
# Using binary cross-entropy loss and Sigmoid activation function gives us an output between $0$ and $1$. If $4$ such neurons are placed in the output layer of a neural network, the network yields $2$ possible outputs for every neuron. Hence we shall have $2^4=16$ outputs, which are more than sufficient to classify $10$ classes.
# 
# We achieve this encoding by converting the class labels in terms of <span style= "color:#E83E8C;">Binary Numbers</span>. For example $9$ can be encoded and re-labelled to $1001$. This allows us the required classification. Keep in mind the binary cross-entropy loss function because when each neuron is concerned it's still a binary classification between $0$ and $1$.
# 
# **The rest of the article is the python code implementing this above idea. Feel free to run this notebook in colab to edit code in real time using the rocket <i class="fa fa-rocket" aria-hidden="true"></i> icon on the top of the page.**

# In[12]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.utils import shuffle


# In[13]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# A look at how the images are present in the dataset. Click on the right "+" icon to see the code.

# In[14]:


fig = plt.figure(figsize=(6, 6))

columns =3
rows = 3
axes = []

# printing 16 training images
for i in range(1, columns*rows +1):
    idx = np.random.randint(1, 100)
    img = X_train[idx]
    axes.append(fig.add_subplot(rows, columns, i))
    subplot_title = y_train[idx]
    axes[-1].set_title(subplot_title)
    plt.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    plt.axis('off')
    
plt.show()


# **The below functions are used to encode image labels to binary number format.**

# In[15]:


def decimalToBinary(n):
    # converting decimal to binary
    # and removing the prefix(0b)
    test = bin(n).replace("0b", "")
    if len(test)<4:
        test = '0'*(4-len(test))+test
    return test

cache_y_train= [decimalToBinary(num) for num in y_train]
cache_y_test = [decimalToBinary(num) for num in y_test]

def final_conversion(array):
    final = []
    for i in array:
        split = []
        for j in i:
            split.append(float(j))
        final.append(np.array(split))
    return np.array(final)

y_train_custom = final_conversion(cache_y_train)
y_test_custom = final_conversion(cache_y_test)

print(len(y_train_custom))
print(len(y_test_custom))


# Data normalization and one-hot encoding for the baseline model with $10$ neurons.

# In[16]:


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train_baseline = np_utils.to_categorical(y_train)
y_test_baseline = np_utils.to_categorical(y_test)


# In[17]:


# train and validation data splits

train_size = int(X_train.shape[0] * 0.9)

train_img, valid_img = X_train[ : train_size], X_train[train_size : ]
train_label_custom, valid_label_custom = y_train_custom[ : train_size], y_train_custom[train_size : ]
train_label_baseline, valid_label_baseline = y_train_baseline[ : train_size], y_train_baseline[train_size : ]


# In[18]:


train_img[0].shape


# In[19]:


def four_neuron_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[20]:


model = four_neuron_model()

# Fit the model
model.fit(train_img, train_label_custom,
          validation_data=(valid_img, valid_label_custom), 
          epochs=5, batch_size=200, verbose = 2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test_custom, verbose=0)
print("\n Four Neuron model Error: %.2f%%" % (100-scores[1]*100))


# In[21]:


baseline = baseline_model()

# Fit the model
baseline.fit(train_img, train_label_baseline,
             validation_data=(valid_img, valid_label_baseline),
             epochs=5, batch_size=200, verbose=2)

# Final evaluation of the model
scores = baseline.evaluate(X_test, y_test_baseline, verbose=0)
print("\n Baseline model Error: %.2f%%" % (100-scores[1]*100))


# In[22]:


#code to re-convert binary numbers to decimals to interpret the results
def decode(array):
    final = ''
    for i in array:
        final+=str(int(i))
    return int(final,2)


# In[ ]:


fig = plt.figure(figsize=(8, 8))

columns = 4
rows = 4
axes = []

np.random.seed(10)
# printing 16 training images
for i in range(1, columns*rows +1):
    idx = np.random.randint(1, 100)
    img = X_test[idx]
    img1 = img.reshape(28,28)
    img_tensor = np.expand_dims(img, axis=0)
    axes.append(fig.add_subplot(rows, columns, i))
    subplot_title = decode(np.round(model.predict(img_tensor)[0]))
    axes[-1].set_title(subplot_title)
    plt.imshow(img1, interpolation='nearest', cmap=plt.get_cmap('gray'))
    plt.axis('off')
    
fig.suptitle('4 neuron model predictions!', fontsize = 16)    
plt.show()

fig2 = plt.figure(figsize=(8, 8))

columns2 = 4
rows2 = 4
axes2 = []

print("\n")
np.random.seed(10)
# printing 16 training images
for i in range(1, columns2*rows2 +1):
    idx = np.random.randint(1, 100)
    img = X_test[idx]
    img1 = img.reshape(28,28)
    img_tensor = np.expand_dims(img, axis=0)
    axes.append(fig2.add_subplot(rows, columns, i))
    subplot_title = np.argmax(baseline.predict(img_tensor))
    axes[-1].set_title(subplot_title)
    plt.imshow(img1, interpolation='nearest', cmap=plt.get_cmap('gray'))
    plt.axis('off')

fig2.suptitle('Baseline model predictions!', fontsize = 16) 
plt.show()


# <table><tr>
# <td> 
#   <p align="center" style="padding: 10px">
#     <img alt="Forwarding" src="data:image/png;base64,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3cxamdk+kn4o6Ylq54aVZ2atzaydpFaSWplZu/xVdqhBfD7rk5n1zn8225vZWZLWkzSiymmVTd0VSuWuoBuk3GH/55J+I+l059wDVc0KpRoiaYGkwZKOycdDqpoRGoPPZ306VrmLsj6R9D1JewVXNte8uryPEgCAcqnHI0oAAMqGQgkAQAKFEgCABAolAAAJycvszYwrfarEOdfQslCNxj6tnkrsU/Zn9fAZrT+F9ilHlAAAJFAoAQBIoFACAJBAoQQAIIFCCQBAAoUSAIAECiUAAAkUSgAAEiiUAAAkUCgBAEigUAIAkEChBAAggUIJAEBC8ukhQFM45phjovbIkSOLel2rVq0qkQ4ARDiiBAAggUIJAEBCzU+9rrPOOj4+7rjjfHzooYdG43bbbbeitjd8+HAfn3XWWVHf559/XkqKWIETTzwxai9btqxKmQDA13FECQBAAoUSAICEmpt6/e53vxu1r776ah/36dOn4OuWLl3aYCxJbdq08fGPfvQjH2evqgz7nHNFZgzUrvDURvbq5IMPPtjHe+yxh4+znw0za7Av/LokjR492se333571DdmzJiVyBooL44oAQBIoFACAJBAoQQAIMFS59rMrGon4tq1a+fjSy+91Menn356NK516/+cZp07d66Ps6u7PPDAAz7+8MMPo74DDzywwfdq27ZtNK5r164+njlzZjL/xnLO2YpHrbxq7tNCnnjiiagdnu9KCc8t14JK7NNK789HH33Ux3vvvXfUV+h8Y6nnKMO+BQsWRH0777yzj6dMmVJU7pXWkj6jKZtuuqmP11577ajvkEMO8XG/fv2ivvA2sJtvvtnHEyZMiMa9/fbb5UizKIX2KUeUAAAkUCgBAEhotreH/OQnP/FxuELOvHnzonGjRo3y8cUXX+zjadOmFdz2KqvE/z8IpwDCqdzFixcXHAe0BOFUWvZz88knn/j4pZde8nH2Vo7wsxzacMMNo3aXLl183KFDh6jvF7/4hY8HDRq0orRRZttss03UPu2003wcroKWnXot1je/+U0ff/XVV1HfG2+84ePx48dHfeHPRfb3dTlxRAkAQAKFEgCABAolAAAJzfYc5d133+3jzTbbzMe/+93vonGlXDq89dZbR+1wGbzQz3/+86g9a9aslX4vrFj2NoHsubBCnn76aR8ffvjhUd+MGTManxh0xRVX+Dh7e8gf//hHH4fnKLP+8Ic/+HjLLbf08S233BKN23333Qtuo7ncElLvevfu7eNTTz3Vx0cccUQ0rlOnTg2+/qOPPora4Wf0vffei/rOOeccH7/44os+3mWXXaJxa621lo/322+/qO+VV17xcXiLSblxRAkAQAKFEgCAhGa7Mk8pVl11VR9nLyHv1auXj7PTCB07dvTxu+++2+BrJGnRokVlybMYLWnVj1JX5gmnaLNPlfn73//e+MTKrBZX5imH8FaPf/zjHz7eaqutonHh76LsVG64Mk9zUQ+f0ez0d7iSTupWj8cff9zHr776qo/PP//8aNzChQsLbuPJJ5/0cfj7evjw4dG47bff3sf//ve/o74ePXr4eN111/Xxp59+WvB9U1iZBwCAElAoAQBIaLZXvZZiwIABPr722muLfl14OH/YYYf5uCmnWtE4Bx10UNRujlOvLcUFF1wQtY866igfb7HFFj7OnvYJ2+HVtmic8AETUny16UknnRT1hVegh9OXN910UzQuvFMgu1pascKVmFq1auXjSy65JBr32GOP+Ti7mlNT4YgSAIAECiUAAAkUSgAAEmruHOWNN94YtY8++mgfr7baaiVtM7wMeqeddvJxuOoDKufss8+O2s8+++xKbyN7jvLMM89sVE5YOTvuuKOPw4efS8U/uDlcwSf7lAiULvvA5PDzlt0H4co64fUa4W09KyM89/iNb3wj6rv11lt9/Mgjj/h4zTXXLLi9bL633Xabj7/44ouSciwGR5QAACRQKAEASKi5qddwqlUqvDhv9tLzsWPH+vj73/9+1BdOD/z+97/3cfYBoiNHjly5ZFGUzz77rNopoJH+9a9/+fj111+P+sKHEKRWAgtXhcl+RsNp2XCB9OxDovF14e83SVq6dGnBseHvvPBhygMHDozGhYvbhxYsWBC1w9WXsisxzZw508fdunUrmFMouzLP0KFDfbxkyZKitlEKjigBAEigUAIAkFBzi6K3bh3PFodTAKln1oXTDTvssEPUN2zYMB/vtddePs5+b8KpoQcffLDIjEtTDwsuF2v11VeP2uHVcNmrWUPhoujhYvaS1LNnzzJlVz4tdVH0cKWecCWY7CorqStiC/Xtu+++0bjwFEul1cpnNHs3wB133OHj/v37R33t27f3cfj5StWJ8Hdrdpq3FMuWLYva4fR69hnB06dPb/T7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/pJmSDpT0j2SPqxiWijdXEmdMl/rJGlOFXIBilLvhZKjyTrgnJvknOvrnOvinNtH0iaS/lHtvFCSNyW1NrPNgq9tJ2lylfIBVqguC6WZ7Sapu7jatS6YWW8za2dm7c3sLEnrSRpR5bRQAufcPEmjJV1qZh3MbHdJB0m6rbqZoVRm1trM2klqJalV/rPautp5lVNdFkrlLuIZ7ZxjOqc+HCtpunLnKr8naS/n3KLqpoRGOEXSasrtzzslDXLOcURZu4YodxveYEnH5OMhVc2ozHh6CAAACfV6RAkAQFlQKAEASKBQAgCQQKEEACAheQmvmXGlT5U45xpauq3R2KfVU4l9yv6sHj6j9afQPuWIEgCABAolAAAJFEoAABIolAAAJFAoAQBIoFACAJBAoQQAIIFCCQBAAoUSAIAECiUAAAkUSgAAEiiUAAAkUCgBAEhIPj0EAIByOPfcc6P2sGHDfPzGG2/4eKuttmqynIrFESUAAAkUSgAAEph6BVBW7du3j9pdunTx8fTp03180kknReMuvPBCH6+77rpR39ChQ3181VVX+Xj+/PmNSxZNxjlXsL1s2bKmTmelcEQJAEAChRIAgASmXgGU1f777x+177zzTh8/+uijPt53330LbiM7TTdkyBAfL1y40Mc33HBDNG7OnDkrlyxQBI4oAQBIoFACAJBAoQQAIKEi5yjDcw/333+/j9u0aVP0NhYsWODjBx98sOC4999/38fXXXedj7/5zW9G42bOnOnj8ePHF50Hmt7222/v48suu8zH++23XzRulVX+8/+87OXl9957r48vuOACH4e3J0jSd7/7XR8//vjjUV/4M4jibb755gX7wn2YPQ950003+fj222+P+iZMmODj8Geia9eu0bhf/vKXK5csKiq8VWjttdeuYiaNwxElAAAJFEoAABIqMvW64YYb+nhlpltDq622mo+POOKIol4TTrtk3zecmnvuueeivnCa7vXXX/fx1KlTo3Hhwr1onHD/9O3bN+r785//7OP11lvPx9mpunCfZvsOO+wwH4dTqN/4xjeicf369fPx8ccfH/WNGjWqYP6I7bjjjj4Op7pTTjnllKg9YsQIHy9evDjq+5//+R8f//jHP/Zx+POB5ie8refMM8+sYiaNwxElAAAJFEoAABIolAAAJFTkHOWf/vQnHy9ZssTHm266aTTugw8+KLiNdu3a+figgw4q6n3DB36us846UV94K8G3vvWtqC/bXi5cKkuSrr76ah9ffPHFReWEhvXp08fHjz32WMFx4e0cp512WtSXenJEeJ583rx5Pr7++uujceG5sOytIyhe+FDetm3bFhwXfg5nzZoV9WXPS4bOOussH++yyy4+Pvzww6Nxf/3rX32cvcUETW/w4ME+zl5HELryyiubIp2ScUQJAEAChRIAgISKTL2G063hNGyprr322qLGbbPNNj7ea6+9Co476qijonZ4aXsonP6VpF/84hc+vuaaa6K+2bNnF5VjS9arVy8fp1ZbClfIOe+883z80ksvFf1e66+/vo8feOABH3fu3DkaF06nZ1fmQfHCabXUFFt4OuOzzz4revvhU0HefPNNH2+77bbRuPB2BKZem174pBhJMjMfZ38uwtuD7rjjjsom1kgcUQIAkEChBAAgoa4e3Pzaa681GGeFiy9LUvfu3X0cXqUVrgAiSZ06dfJxdpWJiy66aOWSbYEuvPBCH4cLJD/88MPRuDPOOMPHb7/9dknvFU7D77DDDgXHpa64Rfn97Gc/8/GTTz5Z0jbuuusuH4crMEnSZpttVlpiKFl4SqV3795RXzjd+tZbb0V9L774oo+XLl1aoezKgyNKAAASKJQAACRQKAEASLDUpdxmVrizToWrB4WXoUvxJerZ817vvvtuWfNwztmKR628ptynf/zjH6P2iSee6ONwtZxdd901Ghc+waVY2afF/O1vf/Pxd77zHR8/9dRT0bg999xzpd+rVJXYp9X8jIa34IRP1gmf/JPVunXjL4sIf15SD2Evx3ul1MNntFThecnwFsCdd945GhfeHnLVVVdFfeGtX81FoX3KESUAAAkUSgAAEurq9pBySC3A3rFjRx8PHDgw6vv1r39dsZxq1U477RS1w2n+uXPn+riUqVYpnm697LLLor499tijwfe99NJLS3ovfF14u1T79u2rkkM4tYemE56iyk631iOOKAEASKBQAgCQwNSrpE022cTHl1xyScFxX375pY+zV3Si8jbaaKOoHS6qHK7mkxU+Z/Kf//xnudNqsX7yk5/4OHX1fCVV631bmnAlLUkaM2ZMUa8Lr25PPX+4ueOIEgCABAolAAAJFEoAABI4RynpwAMP9HGHDh0KjgvPS37++ecVzakeZG/7CB+y26VLFx+//PLLRW0ve54kXBkmda4qfCDzF198UdR7YcWOPPLIBr/+4YcfRu1nn322KdJBBS1YsCBq33fffT4+9NBDC75u5syZPi71NrDmgCNKAAASKJQAACS0yKnXcFUJSRo6dGiD48JLm6V48V+s2EknnRS1w5Vc9ttvPx+HU7IrY8CAAT4+7rjjor7wgb4333xzSdtH2rrrruvjcOo7O9VaaIoWtSP7u3DUqFE+Tk29brjhhj7+/ve/H/VlH1DQnHFECQBAAoUSAIAECiUAAAkt5hxleGvBb37zm6iv0C0hF110UdSeMmVK+ROrY9lLysPbcPr16+fj7FNGQpMnT/bxo48+GvXdeOONPs4+zSV86PY777xTXMJYKdV6ckf4s5PNoZbOe9WS8HYuSRo5cmRRr/vHP/7h4+wTfmoJR5QAACRQKAEASGgxU6+DBw/2cXhbQda7777r4+uuu66iObVk48aNazBeGSeffLKPsyvzPP/88z7+9NNPS9o+0sLveRg//PDDFX3fHXbYocH3lWp7eq85C0+bSPFD7FMeeOABH8+fP7+sOTUljigBAEigUAIAkFC3U6/Z1UB++ctfFhwbrjpx8MEH+3jZsmVlzwulyz64OTR37tyo/dvf/rayyaCgt956q+zbbN++vY832GCDJn1vSOeee25Jrxs2bFiZM6kOjigBAEigUAIAkEChBAAgoa7OUfbt29fHt9xyS9SXWkXkhBNO8PFrr71W9rxQHhdeeGHBvr/+9a9R+6WXXqp0Oigg+9SYcjy4+e677/bxN7/5TR9nb0WZPn16o98LOZdffrmPU9cHhOr1wegcUQIAkEChBAAgoeanXjt37uzjhx56yMeFFjqX4sW0JenBBx8se14oj169evk4fBhz1tixY5siHQQeeeQRH4cP4u7atWs0LvyMhlNz2YW2e/fu7ePsAwnC0yqvvvqqj8PVmSRp6dKlRWSOYnTr1s3Hq666alGv2X///SuVTlVxRAkAQAKFEgCABAolAAAJNXeOcpVV4tp+/PHH+zh1XvLFF1/08RlnnBH1LVmypEzZodz69Onj4+wTC8InRyxcuLDJckLOj3/8Yx+Ht1WF5yslaeLEiT6eNGmSj3fddddoXPfu3Qu+V3heMjwPxu0gaAocUQIAkEChBAAgwbIPPo06zQp3Vsluu+0WtcePH1/U637wgx/4+N577y1rTpXgnCu8lFAjNMd9mhI+9eXqq6+O+iZPnuzj7bbbrslyKlUl9mlz2Z/hA5PPO++8guPCFbJSv3vefPPNqP29733Px81lurXeP6ObbrqpjzfbbLOoL3yayODBg30cfiYlac6cORXKrjIK7VOOKAEASKBQAgCQUBNTr506dfLxe++9F/WtueaaPg6ndZ5++ulo3J577unjr776qtwpll29T+sU6+WXX/bxtttuG/WFUz6/+c1vmiynUtXz1Gvbtm19nD09MmbMGB+HVy6HK2lJ8Uo/d911V9Q3e/bssuRZTnxG6w9TrwAAlIBCCQBAAoUSAICEmliZJ7w0PDwnmRWel/zhD38Y9dXCeUl83euvv+7j7DlKNB+LFi3y8ZNPPhn1hU8PAWoRR5QAACRQKAEASKiJqddw+m3GjBlR31tvveXjo48+2scfffRR5RNDxT322GM+7tmzZ9T3/PPPN3U6AFogjigBAEigUAIAkEChBAAgoSaWsGuJWB6r/tTzEnYtEZ/R+sMSdgAAlIBCCQBAQnLqFQCAlo4jSgAAEiiUAAAkUCgBAEigUAIAkEChBAAggUIJAEDC/wOyhn7cko33KgAAAABJRU5ErkJggg==">
#     <br>
#   </p> 
# </td>
# <td> 
#   <p align="center">
#     <img alt="Routing" src="data:image/png;base64,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JtXxVaC+wqG46667LmoLZ9VJfU7CofrsTEs8VBvVxBElAAAJFEoAABIolAAAJDSaKewaoy233DJa3muvvQr2PffccyudTpPw2Wef+Th8MLcUP2EinHIuO/3cggULfDxx4kQfjx49Oup31113+fjRRx+N2sJzWk899ZSPp02blswfpclOXVaKcKq7hpwKEVgSjigBAEigUAIAkMDQa0Y43Nq/f/+orWPHjj4eO3Zs1DZq1KiK5tUUPfzww9HyE0884ePssHYovOVk3LhxBfutv/76Ps7eThC6//77k3micejUqZOPw9uCgGrjiBIAgAQKJQAACQ069BpOmC1JM2bMaMjNFxReYXfGGWf4+KCDDor6ff7553X2k5rvzDxLI7ya9aWXXqr3+rp06VJUv5dffrne2wJQt+zV7aE2bdr4eM0114zawivkGzuOKAEASKBQAgCQQKEEACChQc9RPvPMM9FyeM6vQ4cOUdvKK6/s43I8WaRHjx4+PvHEE6O2nj17+nirrbYquI5wLJ7zXtV3wAEHVDsFlNEaa6xRsC18+DMal5kzZxZsW3HFFX3cu3fvqK0pPUSdI0oAABIolAAAJDSamXk22mijaDl82OvkyZPrvf5tt93Wx+EMIFnhMG92ZplXXnml3nmgdGuttVa0/Otf/7pg3+eee87H2Yc6o3HacccdC7aFk+OjcXnwwQd9nDp11ZRxRAkAQAKFEgCABAolAAAJVT1HGT7Id+DAgVFbeMtGuS1atChaDh/me/XVV/v48ssvr1gOWHrdunWLlldYYYWCfR966CEfM70ggPrgiBIAgAQKJQAACVUdeh05cqSPszPdhLeHbLLJJvXe1l/+8hcfv/7661HbTTfdVO/1o/I6d+5csG327NnR8nXXXVfpdFBBkyZNipbfeeedKmWCJQlnTXrhhReitnBGtDfffLPBcio3jigBAEigUAIAkNBoZub54osvouXwkB2QpL59+xZse+utt6LlhQsXVjodlFn4YPcbb7wxaktNvI3qGj16tI932GGH6iVSQRxRAgCQQKEEACCBQgkAQII55wo3mhVuREU556wS623K+zQ7o1L4s5u9xeekk05qkJyWRiX2aVPen00dn9HaU2ifckQJAEAChRIAgIRGc3sIsCTLLMPfdQAaHr95AABIoFACAJBAoQQAIIFCCQBAAoUSAIAECiUAAAnJmXkAAGjuOKIEACCBQgkAQAKFEgCABAolAAAJNVkozWy4mU02s+/M7D0zO67aOaE0Znaymb1qZvPMbGi180H98fmsPWa2kZk9bWbTzewDM9uv2jmVU01e9Wpm3SV94JybZ2YbShotaU/n3GvVzQxLy8z2l7RI0m6SlnPOHVXdjFBffD5ri5m1lPSOpJskXSupl6RHJG3hnHuvmrmVS00eUTrnxjvn5i1ezP/rVsWUUCLn3Ajn3IOSvq52LigPPp81Z0NJa0i6xjm30Dn3tKSxkg6vblrlU5OFUpLM7M9mNlvSBEmTJT1e5ZQA5PH5rClW4LVNGjqRSqnZQumcO1FSe0k7SBohaV76HQAaCp/PmjJB0peSzjSzVma2q3LDr22rm1b51GyhlKT8MMAYSWtK6lftfAD8gM9nbXDOLZC0r6Q9JU2RdLqkeyV9VsW0yqpltRNoIC3FORCgseLz2cQ5595U7ihSkmRmL0gaVr2MyqvmjijNrLOZHWxmy5tZCzPbTdKvJT1d7dyw9MyspZm1kdRCUgsza5O/yg5NEJ/P2mRmPfKfzbZmdoak1SUNrXJaZVNzhVK5K+j6KXfY/42kqySd5px7qKpZoVQDJc2RNEDSYfl4YFUzQn3w+axNhyt3UdaXkn4haZfgyuYmrybvowQAoFxq8YgSAICyoVACAJBAoQQAIIFCCQBAQvIyezPjSp8qcc7VNS1UvbFPq6cS+5T9WT18RmtPoX3KESUAAAkUSgAAEiiUAAAkUCgBAEigUAIAkEChBAAggUIJAEAChRIAgAQKJQAACRRKAAASKJQAACRQKAEASKBQAgCQkHx6CNAQDjvssGh52LBhRb2vRYsWlUgHACIcUQIAkEChBAAgockPva6yyio+PuKII3y8//77R/222267otY3ZMgQH59xxhlR2zfffFNKiliCY445JlpetGhRlTIBgP/FESUAAAkUSgAAEprc0OtOO+0ULV955ZU+7tmzZ8H3LVy4sM5Yklq1auXjo48+2sfZqyrDNudckRkDTVd4aiN7dfK+++7r4x122MHH2c+GmdXZFr4uSSNGjPDxHXfcEbWNHDlyKbIGyosjSgAAEiiUAAAkUCgBAEiw1Lk2M6vaibg2bdr4+OKLL/bxaaedFvVr2fKH06wzZ870cXZ2l4ceesjHn332WdS2995717mt1q1bR/06d+7s46lTpybzry/nnC2519Kr5j4t5Omnn46Ww/NdKeG55aagEvu00vvziSee8PGuu+4atRU631jqOcqwbc6cOVHb1ltv7eMJEyYUlXulNafPaMq6667r45VXXjlq22+//Xy84447Rm3hbWA33XSTj8eOHRv1++CDD8qRZlEK7VOOKAEASKBQAgCQ0GhvD/nNb37j43CGnFmzZkX9hg8f7uMLL7zQx5MmTSq47mWWif8+CIcAwqHc+fPnF+wHNAfhUFr2c/Pll1/6eNy4cT7O3soRfpZDa6+9drTcqVMnH7dr1y5qO/XUU33cr1+/JaWNMttkk02i5ZNPPtnH4Sxo2aHXYv3kJz/x8ffffx+1vfvuuz4eM2ZM1Bb+XGR/X5cTR5QAACRQKAEASKBQAgCQ0GjPUd5zzz0+Xm+99Xz8pz/9KepXyqXDG2+8cbQcToMX+u1vfxstT5s2bam3hSXL3iaQPRdWyPPPP+/jAw88MGqbMmVK/RODLrvsMh9nbw/5y1/+4uPwHGXWLbfc4uMNN9zQxzfffHPUb/vtty+4jsZyS0it69Gjh49POukkHx900EFRvw4dOtT5/s8//zxaDj+jEydOjNrOOussH7/22ms+3mabbaJ+K620ko/32GOPqO2NN97wcXiLSblxRAkAQAKFEgCAhEY7M08pll12WR9nLyHv3r27j7PDCO3bt/fxRx99VOd7JGnevHllybMYzWnWj1Jn5gmHaLNPlXnuuefqn1iZNcWZecohvNXjX//6l4832mijqF/4uyg7lBvOzNNY1MJnNDv8Hc6kk7rV46mnnvLxW2+95eNzzz036jd37tyC63jmmWd8HP6+HjJkSNRv88039/F///vfqG2ttdby8Wqrrebjr776quB2U5iZBwCAElAoAQBIaLRXvZZin3328fE111xT9PvCw/m+ffv6uCGHWlE/ffr0iZYb49Brc3HeeedFy4cccoiPN9hgAx9nT/uEy+HVtqif8AETUny16XHHHRe1hVegh8OXN954Y9QvvFMgO1tascKZmFq0aOHjiy66KOr35JNP+jg7m1ND4YgSAIAECiUAAAkUSgAAEprcOcobbrghWj700EN9vNxyy5W0zvAy6K222srH4awPqJwzzzwzWn7ppZeWeh3Zc5Snn356vXLC0tlyyy19HD78XCr+wc3hDD7Zp0SgdNkHJoeft+w+CGfWCa/XCG/rWRrhuccf/ehHUdttt93m48cff9zHK664YsH1ZfO9/fbbffztt9+WlGMxOKIEACCBQgkAQEKTG3oNh1qlwpPzZi89HzVqlI9/+ctfRm3h8MCf//xnH2cfIDps2LClSxZF+frrr6udAurpP//5j4/feeedqC18CEFqJrBwVpjsZzQclg0nSM8+JBr/K/z9JkkLFy4s2Df8nRc+TPmAAw6I+oWT24fmzJkTLYezL2VnYpo6daqPV1111YI5hbIz8wwaNMjHCxYsKGodpeCIEgCABAolAAAJTW5S9JYt49HicAgg9cy6cLhhiy22iNoGDx7s41122cXH2e9NODT08MMPF5lxaWphwuViLb/88tFyeDVc9mrWUDgpejiZvSR169atTNmVT3OdFD2cqSecCSY7y0rqithCbbvvvnvULzzFUmlN5TOavRvgzjvv9HHv3r2jtrZt2/o4/Hyl6kT4uzU7zFuKRYsWRcvh8Hr2GcGTJ0+u9/ZCTIoOAEAJKJQAACRQKAEASGhy5ygrIbzFJLzMffXVV4/6hedawvOaldBUzn+UQ9euXaPl999/v6j3hedQJk6cGLWtu+669c6r3JrrOcpQOAtW+NBdKb4GYP/994/awqeOhOcon3/++ahfdhaaSqqFz2jHjh2j5QEDBvh4++2393H2Fq5PP/3Ux61bt/bxZpttFvXbZpttljqnm266KVoOHwZdydl3JM5RAgBQEgolAAAJDL1mXHDBBT7OPkA0vAWh0kN7tTCsUyyGXkvXGPdnOayyyirR8v/7f//Px4cddpiPs7+/+vXr5+NwNp9KaE6f0WKFt3ZJ8b7KmjFjho/79+/v46FDh0b9UjMJlRtDrwAAlIBCCQBAAoUSAICEJvf0kEpr1apVwbb58+c3YCYAFgufPBGel8yeo8w+uQSVd9ZZZ/n44IMPLvp9J5xwgo/vuuuusuZUbhxRAgCQQKEEACChqkOvJ598so+nT58etd1+++0NnY4k6YgjjijYlr30GUBlZJ8S0bNnTx+HM/NkP69jxoypbGKQFD8FZuDAgT7OPt0pNH78+Gh5xIgR5U+sQjiiBAAggUIJAEBCgw69Zmdg+f3vf+/jf/zjH1FbJYdewxldpPiqrTXWWKPg+8aNG1exnJqzmTNnRstvvfWWj7OTLIfC/fjjH/84agtnWLr44ovrmyIqIDv7zjnnnOPjU089NWoLr26dOnWqj7OToqMyspObhzMlZR+8Hgo/2+FVrpI0b968MmVXeRxRAgCQQKEEACCBQgkAQEKDnqPMnkdaccUVfdyuXbsGy2PTTTeNli+77LI6+919993R8ujRoyuVUrMWnnOS4of2jh071sedO3cuuI5FixaVPzGU3YYbbujj7O0BhR7OLEkTJkzwcffu3SuUHQrZe++9o+X27dvX2W/WrFnR8j777OPj8LPc1HBECQBAAoUSAICEBh16DR98LEnTpk2r2LbCYV1Juvrqq318wAEHFHxfeAvIUUcdFbUxKXrD+Pjjj308d+7cktbx85//3MdHHnmkj4cNG1ZyXihNeKvXvvvu6+O2bdtG/cJbQEaOHBm1pWbMQmWEw6vhLXQpd9xxR7RcK6erOKIEACCBQgkAQAKFEgCABMs++DRqNCvcWAYffPCBjzt27Bi1/e1vf/Nxauq4cBqz7bff3se/+MUvon7rrbeejxcsWBC13XvvvT4+7bTTfPz1118X3G6lOedsyb2WXqX3abk99thjPl5rrbWito033tjHqdtDnn32WR/37t27jNktnUrs04bcn9lbuAo9TSc8DynFn9FwP02aNCnq97vf/c7H2XOUjVEtfkbD6ej+85//+LhLly4F3/Pmm2/6eNttt43aSr3GoFoK7VOOKAEASKBQAgCQUNUHN7/44os+PvTQQ6O2008/vah1hDN4hMPI2VtPbrjhBh9feumlUduUKVOK2hYa3rHHHuvj7NBfeMn6McccU3Adt956a/kTa4YGDBgQLffp08fHhT6HUjzcGs7G069fv6hfdoYmNLydd97Zx2uuuaaPU6fowiHzpjbUWiyOKAEASKBQAgCQUNWrXsOHJB999NFRWzjx8cEHH+zjl19+OeoXPuQ3vEr15ptvjvqFs700BbV4RV1z19Sves1+po477jgfz54928fhBOZS/NCBpnA1a7Fq8TP6xhtv+Dj78IjQlVde6eOzzz67ojk1JK56BQCgBBRKAAASKJQAACRU9RwlCqvF8x/NXVM/R3nYYYdFy+E5ylGjRvl48ODBDZVSVdXiZzScLSm8PeTLL7+M+m2++eY+njx5csXzaiicowQAoAQUSgAAEhh6baRqcVinuWvqQ6+I1eJnNJxlJ3zY/SmnnBL1u/766xssp4bE0CsAACWg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">
#     <br>
#   </p> 
# </td>
# </tr></table>

# ## Results discussion and Inference: Part-1
# 
# The above image shows the performance of 4-neuron-model with binary crossentropy loss as compared to the baseline 10-neuron-model with categorical crossentropy.
# 
# <span style= "color:#E83E8C;">1. What is the reason for reduced accuracy of the 4-neuron-model although 4 neurons are sufficient to encode the labels?</span>
# 
# While the 4 neurons are sufficient to encode all the $10$ labels, the neurons are not sufficient to uniquely represent the features of all the classes. Let's dive a little deep into this inference.
# 
# Consider the first neuron in the output layer. Suppose this neuron learns the features to decide if the number in the image is a $0$. For the sake of intuition lets suppose each neuron is reponsible for ${\frac{1}{4}}^{th}$ of the image. Hence for the number $0$ all the neurons learn their representations. However, if we had 4 outputs, then the first output neuron would be trying to decide what the most significant bit of the digit was. And there’s no easy way to relate that most significant bit to simple shapes like those shown above.
# 
# <p style="text-align:center"><img src="images/mnist_zero.PNG"/></p>
# 
# In other words, there are no sufficient neurons to capture the representations (features) of all the image labels individually. All the neurons work simultaneously for every image class and thus have a poorer representation as a whole as compared to 10 neurons with softmax activation (where each class is uniquely represented by each neuron in terms of probability.)

# ## Extending the 4-neuron-model to 16 classes
# 
# After implementing a 4-neuron-model for $10$ classes, let's extend the code to work for $16$ classes. One thing to keep in mind before proceeding with the code is the size of the images. All the images must be of the same size during training. This can be done using image processing, but I felt it would be the best find a dataset with the same dimensions as MNIST.
# 
# Luckily, there is the **fashion-mnist dataset** with $28 \times 28$ images. Hence, we can combine the two datasets for $16$ classes

# In[23]:


#Loading MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Loading Fashion MNIST dataset
(X_train1, y_train1), (X_test1, y_test1) = fashion_mnist.load_data()


# In[24]:


fashion_x_train = []
fashion_y_train = []

for i in range(len(y_train1)):
    if y_train1[i] in [0,1,2,3,4,5]:
        fashion_x_train.append(X_train1[i])
        fashion_y_train.append(y_train1[i]+10)
        
fashion_x_train = np.array(fashion_x_train)
fashion_y_train = np.array(fashion_y_train)

print(fashion_x_train.shape)
print(fashion_y_train.shape)

fashion_x_test = []
fashion_y_test = []

for i in range(len(y_test1)):
    if y_test1[i] in [0,1,2,3,4,5]:
        fashion_x_test.append(X_test1[i])
        fashion_y_test.append(y_test1[i]+10)
        
fashion_x_test = np.array(fashion_x_test)
fashion_y_test = np.array(fashion_y_test)

print(fashion_x_test.shape)
print(fashion_y_test.shape)


# In[25]:


train_array_x = np.concatenate((X_train, fashion_x_train), axis = 0)
train_array_y = np.concatenate((y_train, fashion_y_train), axis = 0)

test_array_x = np.concatenate((X_test, fashion_x_test), axis = 0)
test_array_y = np.concatenate((y_test, fashion_y_test), axis = 0)

final_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9,
              10: 'T-Shirt',11: 'Trouser', 12: 'Pullover',
              13: 'Dress', 14: 'Coat', 15: 'Sandal'}


# In[26]:


train_array_x, train_array_y = shuffle(train_array_x, train_array_y, random_state=0)
test_array_x, test_array_y = shuffle(test_array_x, test_array_y, random_state=0)


# In[27]:


len(train_array_x)


# In[28]:


fig = plt.figure(figsize=(6,6))

columns =3
rows = 3
axes = []

# printing 16 training images
for i in range(1, columns*rows +1):
    idx = np.random.randint(1,100 )
    img = train_array_x[idx]
    axes.append(fig.add_subplot(rows, columns, i))
    subplot_title = final_dict[train_array_y[idx]]
    axes[-1].set_title(subplot_title)
    plt.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    plt.axis('off')
    
plt.show()


# In[29]:


def decimalToBinary(n):
    # converting decimal to binary
    # and removing the prefix(0b)
    test = bin(n).replace("0b", "")
    if len(test)<4:
        test = '0'*(4-len(test))+test
    return test

cache_y_train= [decimalToBinary(num) for num in train_array_y]
cache_y_test = [decimalToBinary(num) for num in test_array_y]

def final_conversion(array):
    final = []
    for i in array:
        split = []
        for j in i:
            split.append(float(j))
        final.append(np.array(split))
    return np.array(final)


# In[30]:


num_pixels = train_array_x.shape[1] * train_array_x.shape[2]

train_array_x = train_array_x.reshape((train_array_x.shape[0], num_pixels)).astype('float32')
test_array_x = test_array_x.reshape((test_array_x.shape[0], num_pixels)).astype('float32')

train_array_x = train_array_x / 255
test_array_x = test_array_x / 255

# one hot encode outputs
y_train_baseline = np_utils.to_categorical(train_array_y)
y_test_baseline = np_utils.to_categorical(test_array_y)

y_train_custom = final_conversion(cache_y_train)
y_test_custom = final_conversion(cache_y_test)

print(len(y_train_custom))
print(len(y_test_custom))


# In[31]:


train_size = int(train_array_x.shape[0] * 0.9)

train_img, valid_img = train_array_x[ : train_size], train_array_x[train_size : ]
train_label_custom, valid_label_custom = y_train_custom[ : train_size], y_train_custom[train_size : ]
train_label_baseline, valid_label_baseline = y_train_baseline[ : train_size], y_train_baseline[train_size : ]


# In[32]:


train_img[0].shape


# In[33]:


def four_neuron_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[34]:


import time
start_time = time.time()

model = four_neuron_model()

# Fit the model
history_four_nuerons = model.fit(train_img, train_label_custom,
                                 validation_data=(valid_img, valid_label_custom),
                                 epochs=5, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(test_array_x, y_test_custom, verbose=0)
print("\nFour Neuron model Error: %.2f%%" % (100-scores[1]*100))

end_time = time.time()

print("Time taken for training: {} seconds".format(round(end_time-start_time,2)))


# In[35]:


import time
start_time = time.time()

baseline = baseline_model()

# Fit the model
history_baseline = baseline.fit(train_img, train_label_baseline,
                                validation_data=(valid_img, valid_label_baseline),
                                epochs=5, batch_size=200, verbose=2)

# Final evaluation of the model
scores = baseline.evaluate(test_array_x, y_test_baseline, verbose=0)
print("\nBaseline model Error: %.2f%%" % (100-scores[1]*100))

end_time = time.time()

print("Time taken for training: ", round(end_time-start_time,2))


# In[ ]:


fig = plt.figure(figsize=(8, 8))

columns = 4
rows = 4
axes = []

np.random.seed(10)
# printing 16 training images
for i in range(1, columns*rows +1):
    
    idx = np.random.randint(1, 100)
    img = test_array_x[idx]
    img1 = img.reshape(28,28)
    img_tensor = np.expand_dims(img, axis=0)
    axes.append(fig.add_subplot(rows, columns, i))
    subplot_title = final_dict[decode(np.round(model.predict(img_tensor)[0]))]
    axes[-1].set_title(subplot_title)
    plt.imshow(img1, interpolation='nearest', cmap=plt.get_cmap('gray'))
    plt.axis('off')
    
fig.suptitle('4 neuron model predictions!', fontsize = 16)    
plt.show()

fig2 = plt.figure(figsize=(8, 8))

columns2 = 4
rows2 = 4
axes2 = []

print("\n")

# printing 16 training images
np.random.seed(10)
for i in range(1, columns2*rows2 +1):
    idx = np.random.randint(1, 100)
    img = test_array_x[idx]
    img1 = img.reshape(28,28)
    img_tensor = np.expand_dims(img, axis=0)
    axes.append(fig2.add_subplot(rows, columns, i))
    subplot_title = final_dict[np.argmax(baseline.predict(img_tensor))]
    axes[-1].set_title(subplot_title)
    plt.imshow(img1, interpolation='nearest', cmap=plt.get_cmap('gray'))
    plt.axis('off')

fig2.suptitle('Baseline model predictions!', fontsize = 16) 
plt.show()


# <table><tr>
# <td> 
#   <p align="center" style="padding: 10px">
#     <img alt="Forwarding" src="data:image/png;base64,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hL9X165dbXzVVVfZeMOGDU69q6++2sZffvllwudPRr6lXps0aWLjO+64w8YnnniiU0+nwY855hgbL1myJI2tS798fI8mo2bNmjb+5ptvnLJmzZrZeMiQIU7ZLbfcktZ2JYOpVyIioiSwoyQiIgpgR0lERBSQEyvz3HbbbTbWY5KAu1qO//gxZZfTTjut0K/7K/PosUh/mk+iJkyYYONnnnnGxh999JFT7+6777bxCSeckNT3Ki3+9re/OcdXXnmljZs2bRr3dXqM8ogjjrDx22+/nfD37t27t43LlStn4xYtWsR9zT/+8Q/neNu2bQl/P0pc7dq1bRz6O8hlvKMkIiIKYEdJREQUkLWp1/bt29u4f//+NvYfKddTQnbu3Jn+hlHC9HUD3AXst2/fbmP/miabbo1HL4o+atQop0wvpH/77bc7ZXfddVdK25EL9t9/f+f4xRdftPE555yT1DnvvPNOG+vUt/+9dD2dagWA5s2bJ/S9dJr3L3/5i1P20ksvxS2j5FWtWjWhehs3bkxzS9KHd5REREQB7CiJiIgC2FESEREFZM0YZfny5Z3j4cOH27hy5co29se9Zs6cmd6GUZHo5az8aR979uyxsR7/mzJlSvobVsj3BdwpK/7OIqVljFK/95599lmn7Oyzz7bxpk2bnDI9HevBBx+Me35/t54CFStWdI71zkBz5sxxyvTOPVq1atWc4+OOO87G7dq1c8r0UoYLFiyw8cMPP1zouSkx/vs8Hv1cQq7hHSUREVEAO0oiIqKArEm96k2XAXdngt27d9v4+++/L7E2UdH169fPxnqnCN+8efNKojl7pTeJ9jfq1htN6xWg8o3exeG8886LW2/YsGHO8a233lqs76s3SQfir9yUrGOPPdY51tNDBg4caGOmXotHT8nRsX9ct27dEmtTqvGOkoiIKIAdJRERUUDWpF537NjhHOu0TNmyvzZz6tSpJdUkSkIofbZ06VIbjx49uiSaUyT77befc6yf4M03DRo0sLHeHNs3duxYGxc31VrS/I249Wbu/ibClDxjTKFxqF6u4R0lERFRADtKIiKiAHaUREREAVkzRnnBBRc4x82aNbOxHk/I5dUdSoPQo+Kff/65jbNxE12/vXqjYr3riD+enov+/ve/21jv/jBy5Einnr+LR77wrzVRCO8oiYiIAthREhERBWQ09brPPr/206effnrcenPnzi2J5lAKhB4VL8nFz5Pht7dz586FxhMmTCixNqXLiSeeaGP9c/up13yiF3jP5akK2SbRjZsnTZqU5pakD+8oiYiIAthREhERBWQ09apX3NELUPvGjBlT7O9Vrlw5G59yyilO2aBBg2zcqlWruOf4+uuvbfz73//eKdMLt5cmBx10kHPcvn37uHVfffXVdDeHEqRX3LnwwgttPGvWrEw0Jy0OOOAA51jvd7l169aSbk7e0ilt3/Tp0238wQcflERz0oJ3lERERAHsKImIiALYURIREQVkzco8qdalSxfnWK/uox/1B4CJEyfa+LbbbrOxP9525ZVX2vivf/2rU/bQQw8l39gcVqlSpeCxtmzZsnQ3hxK0YMGCQr+up40A7hhTrvF3RdHPKdx9990l3Zy8FVqN65///KeNs3E1rkTxjpKIiCiAHSUREVFARlOvekrFW2+95ZSdddZZNu7Tp4+NQynOSy65xMYPPPCAU7Z27VobX3XVVU7Z8OHDCz1fo0aNnGOdetULZgPAo48+auPSNFVk165dcY+HDRtW0s0pssqVK9vYTxvt3Lmz0DgfPP300zbWwwiHHHJIJpqTMk899ZSNr7jiCqdsw4YNhdaj4gmtxvXmm2+WdHPSgneUREREAewoiYiIAthREhERBWR0jHLPnj02Hjx4sFN2xhln2Lhp06Zxz9G3b18bP/PMMzaePXu2U69Hjx42XrRoUULt02OSvvnz5zvHpXU3gsmTJzvHeoeQbPydnHnmmc7xgAEDbOy39/3337dxtu98UlRLly618Ysvvmhj/X4CgCOPPNLGegnHTKpTp46N9XQuwB2X3LJli1P2xhtv2JhTlUqGHhfOZbyjJCIiCmBHSUREFCCh9JiIZCx39umnn9r4qKOOsrG/qo5+/LhNmzY29jeC1rslhDRo0MDGF198sVOmUzkzZsxwyj777LOEzp8oY4zsvVbRpfuafvnllzZu0qSJU6ZXfSnJzbhr1qxp4/HjxztlrVu3trG/csjJJ59s41Rs1pyOa5qK69muXTsb+zs8rFy50sb+tCo/7Z5K+++/v3PcvHlzG+sUqm47APz000827t+/v1P27rvvprKJOfseTTX9N6Lfa4D7GfrSSy+VWJuSFe+a8o6SiIgogB0lERFRQNamXi+//HIb61U09Ao7AFCjRg0bP/HEEzb2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a2N/CbtDDjnExt9++61TFm+pumzcSYR3lERERAHsKImIiAJKNPWqp3IAbtrU35BZr4qjV13RqVEAWLRokY31o83+xr2hR471ivnvvvuujTt37hy3Tc2aNYt7PkqPgQMHOsft27e3sf47AIDbbrutRNpUmunf/5/+9Ccb65W0AHcFHn9DdcouOsWqV07zVzPTwyB6WGvatGlOvZNPPtnG/m5M8aaf+DuaZAPeURIREQWwoyQiIgoo0dTr5Zdf7hzr23f/KVK9sLJ+Ourtt9926jVu3NjGesWd6tWrO/X0E7d+mk6fXy/a7G8sq8954oknxj3/7t27Qannb7it/e9//3OOZ8+ene7mlHo6va1XRvLTbzNmzCixNlHx6M8//QSsP2NBp031amb+Z6YeAvNnEWg6LeunebMB7yiJiIgC2FESEREFsKMkIiIKKNExyrPPPts51isw6EeRAXfM8scff7TxYYcd5tTTu4To8Up/Ooh+LN3fgUTn4nv16mVjf5eRr7/+2sb+bhWHHnpoofWoePS4cMWKFZ0yvRrP0qVLS6xNpZW/6soJJ5xQaL177703qfPr6V0HHnigU6ann8yaNcvGb775plOP00+KZ+PGjYV+XV8b4Le7MxVYvny5c6zHG/0dlzZs2BD3/NmGd5REREQB7CiJiIgCSjT1qhdBB9xHjv1Hh/XqDHrTZT1tBHAfP540aZKN9QLmAHDdddfZ2J86oKeHbNu2rdAYcBdT12kDwE0JM/WaOno1J39B/LVr19r4iSeeKLE2lVbnnHOOc6xX09KLa/vp0Hjq1KnjHF9yySU2vu+++xI6x/XXX+8cH3zwwQm9jgqnV1XS0/fipVoBdwgkNPUu0ekh2Yh3lERERAHsKImIiALSnno98sgjbVytWjWnbP369TbWC44D7tNrOgX63//+16l36qmn2lg/VTVlyhSnnj5/69atnbL33nvPxk2aNLFxixYtnHqVK1e2sb+XWvfu3W08bNgwUPJ0Sv7mm2+2sb/n5O23327j+fPnp79hpVxoI4BE062av3D9n//8Zxv7T6/ecMMNNj7//PNtrPckBYABAwbY+IEHHihym0o7PZyhh5r8mQKaTqn6q+roxc5DT86G0rLZILtbR0RElGHsKImIiALYURIREQWkfYxyzpw5Nu7Xr59TpscG/THKkSNH2rhevXo2XrVqlVNP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WQngRAAnG2PyZhoYdw8hIiIKyMc7SiIiopRhR0lERBTAjpKIiCiAHSUREVFAcKshEeGTPhlijEnLvD9e08xJxzXl9cwcvkfzT7xryjtKIiKiAHaUREREAewoiYiIAthREhERBbCjJCIiCmBHSUREFMCOkoiIKIAdJRERUQA7SiIiogB2lERERAHsKImIiALYURIREQWwoyQiIgoI7h5S2nXv3t05HjJkiI3r16/vlPXv39/Gb775ZnobRkREJYZ3lERERAHsKImIiAKYegXQo0cPG99888027tixo1OvatWqcc9Ro0aN1DeMiAAAv/vd72z8zjvv2Hj27NlOPf89S5QKvKMkIiIKYEdJREQUkFep16ZNm9p43333dcratWtn44EDBzplhx56qI332Sf+vx3WrVtn4wceeMApe+aZZ4rWWCKKq1y5cs7xgw8+aGP93vafPidKB95REhERBbCjJCIiCmBHSUREFJDzY5S1a9e28fjx423csGHDYp97xIgRzvHgwYNt/OOPPxb7/BTxx4WNMYXGZcu6f667d++Oe85KlSrZ+Pjjj7dxnTp1nHrjxo2zcSquqf+z7Nmzp9jnLI169uzpHB944IGF1rv88stLojmlkp4Od9FFFzllt912m41r1apl49Dfvx5nBoD77rvPxhs2bCheY9OMd5REREQB7CiJiIgCRKe2flMoEr8wQ/xb+5dfftnG5557bkLn2Lhxo3P80ksv2fjee++18erVq516O3fuTLidxWWMkXScNxuvaYiIFBoDblrn1FNPdcoWLlxoY716y/XXX+/UO+mkk2zsL4JfoUIFG1evXt3GmzZtcuqFUsBaOq5prl3PRE2ZMsU57tSpk41nzpxp4w4dOjj1Qp9nqZYP71F/KKJPnz42vvbaa23cvHnzhM7nv0dD10MPbfXt2zeh86dbvGvKO0oiIqIAdpREREQB7CiJiIgCcm56SJUqVZxjPS6p8+P+48bPPvusjf/5z386ZXo8izLPH+co4E+1qFixoo1/+eWXuOd4+OGHbfzkk0869fRyaPvvv79Tpv+22rRpY+OVK1c69bZv327jyZMnO2UzZsxAaaCnaflTNvS0gF27dsU9h15KUi85GVKSY5L54qijjrLxsGHDnDI9FpzM73bMmDHOsX6PHn300U6ZXn7wlFNOiXvO7777zsbLli0rcptSgXeU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">
#     <br>
#   </p> 
# </td>
# <td> 
#   <p align="center">
#     <img alt="Routing" src="data:image/png;base64,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Xbp0sfHbb7+d/opRUPv27W380UcfOWUNGza08WOPPeaUffLJJzaeMWNGmmqX//SdiXrC7GTpuxRnzZpV7OPlu1C3QZs2bZzt9evX21inKP3FlP0UaDx6P/9OVF2mv0/91Kjez38vOgWq06t+ffXz9KxCgPue9V3A9erVc/bTMw4x9UpERJRibCiJiIgC2FASEREFlJo+yg4dOtg4kZw0ZYe+ffva2F+0NeTKK6+08Y033pjSOuWD3r1727hbt2429hfo1X2UnTp1SujYoT4g3Uc5e/ZsZ78RI0bYWK+EQYWrWbOms637EfV584eA6H4+fzUO3R+oz1ui35m6r3FPdH+mfl1/iIqejccffrJ48eJCj1erVi1nP91Hmcz3P68oiYiIAthQEhERBZSa1CvlpoceesjG/q3nF154Ydznvf7662mrUy7SM1MBbkq7fPnycZ/3xBNP2Pizzz5L6LVCqdcePXrYWE9iDQCHHnqojb/77junTC/KSxE/DaknONcTlfvDSPSMOCtWrHDK9CTmoRlxdJpXx/4QkNAwFf17oo+xZs0aZz89ZM+f4F2nXvX7b926tbPf1KlTbczUKxERUYqxoSQiIgpgQ0lERBSQt32Uui9kT/TtyFOmTElHdShJevq5RFc2AID58+enozo565577nG2dV/SuHHjbHzqqac6++kpwlJB94326dPHKdMrwAwdOtQpO/nkk22sV40pbfSwD70yB+AOA9HDI/z99Cob1apVc8p0P6Luo/T79fRr6b7G0PR4/vRzfl92gc2bNzvbS5cutbFe3QlwF4rWr123bt249UgGryiJiIgC2FASEREF5G3qNdFZRADgvffeszFXNMheenUAn79CiJ++Ke3OPffcuGV62EeqU62+M88808YXX3yxUxaaCebyyy+3cWlOveqZk/yUqk6bNmrUyMaffvqps9/atWtt7H9P6tmSdIrWn3En3hAL//F4Q0CA+LPxVK9e3dnv5ptvtvEHH3zglOk6rly50sb+zDzFxStKIiKiADaUREREAXmVetWX25dddlnCz3v33XfTUR1KAZ26iXeXHAB8/fXXzva6devSVqdcVJKLkPtpXn0n7RlnnGHjSpUqxT2GTqMBwLPPPpuayuU4PUG4nw7Vd8Q2adLExnqWHsBd1PyII45wyvQxQ3e96jRqaNHl0DF06lXv50/iru9m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NaIiXxvK+wBMQPTX648AvgfQP6M1ouK4BtFt58sAvA7gag4lyGk8n/mlH6Iur1sAXBSL8+qekLwcHkJERJQq+XpFSURElBJsKImIiALYUBIREQWwoSQiIgoI3sIrIrzTJ0OMMZKO4/KcZk46zinPZ+bwM5p/4p1TXlESEREFsKEkIiIKYENJREQUwIaSiIgogA0lERFRABtKIiKiADaUREREAWwoiYiIAthQEhERBeTV4pplypSxccuWLZ2ytm3b2rhfP3eptEMPPdTG06dPt/Ho0aOd/V566SUbT548uXiVJSKinMArSiIiogA2lERERAFsKImIiAJyvo+yW7duNtZ9j927d0/4GN9++62NFy1aZONevXo5+3322Wc2Zh9l0T366KM2Pu+882x85plnOvvNmTPHxqNGjbJx586dnf1mz55t4+HDhztlTzzxhI1Xrlxp402bNhWx1kRU2vGKkoiIKIANJRERUYAYE3+N0GxcQLRDhw7O9siRI21cq1YtGy9YsMDZT6f6li5d6pQtX77cxhs3brTxAw884Oz32muv2Xjq1KlFqXaR5eOisDodrs+b7z//+Y+Ne/ToYePQ76pPZPeP76uvvrLx5Zdf7uw3a9ashI9ZXKVl4ebq1avHLVu9enUJ1iS98vEzWlxjxoyJu/3CCy84ZfPmzSuJKhUJF24mIiJKAhtKIiKigJxIve611+72/K233nLKzjjjDBtfd911Nn7mmWfSXq90yse0Tt++fW384IMPJvSca6+91sbff/99wq/VtWtXG1999dU2Xrt2rbPfNddcY+Nx48YlfPxk5FvqtUmTJja+6667bHzcccc5++k0+JFHHmnjhQsXprF26ZePn9Fk1KxZ08YTJkxwypo1a2bjAQMGOGW33nprWuuVDKZeiYiIksCGkoiIKIANJRERUUBOzMxzxx132Fj3SQLubDn+7ceUXU455ZRCH/dn5tF9kf4wn0SNHz/exi+++KKNP/nkE2e/++67z8bHHntsUq9VWvzf//2fs33VVVfZuGnTpnGfp/soDzvsMBu/9957Cb/2ueeea+Ny5crZuEWLFnGf8/e//93Z3rx5c8KvR4mrXbu2jUO/B7mMV5REREQBbCiJiIgCsjb12r59exv36dPHxv4t5XpIyLZt29JfMUqYPm+AO4H9li1bbOyf02TTrfHoSdGHDRvmlOmJ9O+8806n7N57701pPXLBvvvu62y/8sorNj7nnHOSOuY999xjY5369l9L76dTrQDQvHnzhF5Lp3n/9re/OWWvvvpq3DJKXtWqVRPab926dWmuSfrwipKIiCiADSUREVEAG0oiIqKArOmjLF++vLM9aNAgG1euXNnGfr/XtGnT0lsxKhI9nZU/7GPXrl021v1/EydOTH/FCnldwB2y4q8sUlr6KPVn76WXXnLKzj77bBuvX7/eKdPDsR555JG4x/dX6ylQsWJFZ1uvDDRz5kynTK/co1WrVs3ZPvroo23crl07p0xPZTh37lwbP/7444UemxLjf87j0fcl5BpeURIREQWwoSQiIgrImtSrXnQZcFcm2LFjh41//PHHEqsTFV2vXr1srFeK8M2ePbskqrNHepFof6FuvdC0ngEq3+hVHM4///y4+w0cONDZvu2224r1unqRdCD+zE3JOuqoo5xtPTzk9ttvtzFTr8Wjh+To2N+uW7duidUp1XhFSUREFMCGkoiIKCBrUq9bt251tnVapmzZ3dWcNGlSSVWJkhBKny1atMjGw4cPL4nqFMk+++zjbOs7ePNNgwYNbKwXx/aNGjXKxsVNtZY0fyFuvZi7v4gwJc8YU2gc2i/X8IqSiIgogA0lERFRABtKIiKigKzpo7zwwgud7WbNmtlY9yfk8uwOpUHoVvHPP//cxtm4iK5fX71QsV51xO9Pz0UPPfSQjfXqD++8846zn7+KR77wzzVRCK8oiYiIAthQEhERBWQ09brXXrvb6VNPPTXufrNmzSqJ6lAKhG4VL8nJz5Ph17dz586FxuPHjy+xOqXLcccdZ2P9vv3Uaz7RE7zn8lCFbJPows3ffvttmmuSPryiJCIiCmBDSUREFJDR1KuecUdPQO0bOXJksV+rXLlyNj7xxBOdsn79+tm4VatWcY/x9ddf2/gPf/iDU6Ynbi9NDjjgAGe7ffv2cfd944030l0dSpCeceeiiy6y8fTp0zNRnbTYb7/9nG293uWmTZtKujp5S6e0fVOmTLHxRx99VBLVSQteURIREQWwoSQiIgpgQ0lERBSQNTPzpFqXLl2cbT27j77VHwC++eYbG99xxx029vvbrrrqKhtfd911Ttmjjz6afGVzWKVKlYLb2uLFi9NdHUrQ3LlzC31cDxsB3D6mXOOviqLvU7jvvvtKujp5KzQb19NPP23jbJyNK1G8oiQiIgpgQ0lERBSQ0dSrHlLx7rvvOmVnnXWWjXv06GHjUIrz0ksvtfHDDz/slK1atcrGV199tVM2aNCgQo/XqFEjZ1unXvWE2QDw5JNP2rg0DRXZvn173O2BAweWdHWKrHLlyjb200bbtm0rNM4HL7zwgo11N8JBBx2UieqkzPPPP2/jK6+80ilbu3ZtoftR8YRm43r77bdLujppwStKIiKiADaUREREAWwoiYiIAjLaR7lr1y4b33333U7ZaaedZuOmTZvGPUbPnj1t/OKLL9p4xowZzn4nn3yyjefPn59Q/XSfpG/OnDnOdmldjeC7775ztvUKIdn4MznzzDOd7b59+9rYr++HH35o42xf+aSoFi1aZONXXnnFxvrzBACHH364jfUUjplUp04dG+vhXIDbL7lx40an7K233rIxhyqVDN0vnMt4RUlERBTAhpKIiChAQukxEclY7mz06NE2PuKII2zsz6qjbz9u06aNjf2FoPVqCSENGjSw8SWXXOKU6VTO1KlTnbLPPvssoeMnyhgje96r6NJ9TseNG2fjJk2aOGV61peSXIy7Zs2aNh47dqxT1rp1axv7M4eccMIJNk7FYs3pOKepOJ/t2rWzsb/Cw7Jly2zsD6vy0+6ptO+++zrbzZs3t7FOoeq6A8Avv/xi4z59+jhlH3zwQSqrmLOf0VTTvyP6swa436GvvvpqidUpWfHOKa8oiYiIAthQEhERBWRt6vWKK66wsZ5FQ8+wAwA1atSw8bPPPmtjf0LkROkU7SOPPOKUffLJJ0kdMxm5mtbRsyP985//dMpef/11G1988cXprIYz487w4cNtfOSRR8Z9zsKFC53t0N3WycjW1Kt20003Odt6MYFp06Y5ZXrGrGQWfC5fvryzfc4559jYXxg93uLAP//8s7N966232jjds8Lk6mc01UKpV92VtXTp0hKrU7KYeiUiIkoCG0oiIqIANpREREQBWbtw87/+9S8bH3/88TY+99xz4z5Hz/QTWkDY7/+47bbbbKxvN3/nnXcSqyxZeiWWY4891in74x//aOPu3bvb+KuvvnL227p1a0KvVaFCBRvroRyAO2OLP6RI++KLL2ysZ+kprfwVX/Ti5f6sPWPGjCn0GP6wEb0qi78wdKLWr19faB31Z5cyQ5/f2bNnO2VVq1a1cS70UcbDK0oiIqIANpREREQBWTs8RLvgggtsPGTIkISeM2nSJGdbpwc6duzolK1Zs8bGN954o431ZNEAsHPnzoReOxXy4dbz/fbbz9nWsy3Vr1/fxr169XL207MeValSxcaXX365s59eWFtP3g2451v/jvszKv31r3+1sZ5VKB1yYXhIiD+hvB7Ocfrpp9tYp8SB+OfCp4d6PPfcc07ZG2+8YWN/GE+m5MNnNBVCw0N0t8eUKVNKrE7J4vAQIiKiJLChJCIiCmBDSUREFJATfZR77bW7Pe/Xr59TdtdddxX5eP4CtLfffruNU70KSLLysf/jgAMOsLFeFFn3NfoS7d8KPU8PP9FT7AG/XYA7nXK9jzJE90eXK1cuqWPMnTvXxno4SLbKx89oMkJ9lHoqUj10LFuxj5KIiCgJbCiJiIgCciL1Whrle1pHp2g6dOjglOlZdbp27Wrj0O+qv1LJxx9/bOORI0fa2F+cuSTlc+q1NMr3z2ii9Gw8LVq0cMr0ajSPP/54idUpWUy9EhERJYENJRERUQBTr1mKaZ38w9RrfuFnNKJnOuvdu7dT9tNPP9n4oYceKrE6JYupVyIioiSwoSQiIgpgQ0lERBTAPsosxf6P/MM+yvzCz2j+YR8lERFREthQEhERBbChJCIiCmBDSUREFMCGkoiIKIANJRERUQAbSiIiogA2lERERAFsKImIiAKCM/MQERGVdryiJCIiCmBDSUREFMCGkoiIKIANJRERUUBWN5QiMk9ENovIehFZIyJfiUgvEcnqelPqiEhvEflWRLaKyOBM14dSQ0R6iMiPIrJRRH4SkWMyXafSREQ2qH+7Yt+zBdsXZrp+2aZspiuQgD8YY0aJSFUAXQE8CeBwAJf6O4pIGWPMzpKuIKXVIgD3AzgJQPkM14VSQEROADAAwPkAvgFQP7M1Kn2MMZUKYhGZB+AKY8wofz8RKWuM2VGSdcvGOuTMlZkxZq0xZiiiD9efReQAERksIv8QkY9EZCOA7iLSQETeEZHlIjJXRP5WcAwROSx2dbJORJaKyGOxx8uJyBARWRm7cp0gInUz9FZJMca8a4x5H8DKTNeFUuYeAPcaY8YbY3YZYxYaYxZmulIEiEg3EVkgIv8nIksADBKRfUXkCRFZFPv3hIjsG9u/p4iM9Y5hRKRVLD5FRKbHsoILReQmtd9pIjJJZQs7qLJ5sTpMAbBRRDJ6UZczDWUBY8w3ABYAKEjV/AlAfwCVAXwFYBiAyQAaAjgOwPUiclJs3ycBPGmMqQKgJYA3Y4//GUBVAI0B1ATQC8DmtL8ZolJGRMoAOARAbRGZE/tSfkZEmC3IHvUA1ADQFMBfANwO4AgAnQB0BHAYgH4JHuslAFcZYyoDOADAfwFARA4C8C8AVyH6zn0ewNCCBjjmAgCnAqjGK8rkLEJ0IgHgA2PMOGPMLgAHAqhtjLnXGLPNGPMzgH8C6BHbdzuAViJSyxizwRgzXj1eE0ArY8xOY8x3xph1Jfh+iEqLugD2BnAOoj92OwHojMS/eCn9dgG4yxiz1RizGcCFiDIAy4wxyxFlBC5O8FjbAbQTkSrGmNXGmImxx68E8Lwx5uvYd+7LALYiapALPGWMmR+rQ0blakPZEMCqWDxfPd4UQIPYpfwaEVkD4DZEH04AuBzAfgBmxNKrp8UefwXACABvxFILfxeRvdP+LohKn4IvvaeNMYuNMSsAPAbglAzWiVzLjTFb1HYDAL+o7V9ijyXibETn9hcR+UJEfhd7vCmAG73v6sbececjS+RcQykihyJqKAvy4noOvvkA5hpjqql/lY0xpwCAMWa2MeYCAHUQ3UzwtohUNMZsN8bcY4xpB+BIAKcBuKTE3hRRKWGMWY2o64RzZ2Yv/9wsQtSwFWgSewwANgKoUFAgIvWcAxkzwRjzR0Tfue9jd3fXfAD9ve/qCsaY1wP1yJicaShFpErsCvANAEOMMT8Usts3ANbFOoHLi0iZ2E0/h8aOcZGI1I6ladfEnrNTRLqLyIGx/pN1iNIFvHs2C4hIWREpB6AMgDKxG69y4W5tim8QgGtFpI6IVAdwPYAPM1slCngdQD8RqS0itQDcCWBIrGwygPYi0in2Ob274Ekiso+IXCgiVY0x2xF9txZ8r/4TQC8ROVwiFUXkVBGpXGLvqghyoaEcJiLrEf0FcjuiNM1vhoYAQGxoyB8Q9XvMBbACwIuIbtQBgJMBTBORDYhu7OkRSzHUA/A2ohP5I4AvsPsXgTKrH6J03S0ALorF7M/KbfcBmABgFqLP2/eIbsij7HQ/gG8BTAHwA4CJscdgjJkF4F4AowDMxu5MX4GLAcwTkXWIbpK8KPa8bxH1Uz4DYDWAOQB6pvl9JI2rhxAREQXkwhUlERFRxrChJCIiCmBDSUREFMCGkoiIKCB4m72IpPROHxFxtkM3EjVs2NDGq1evtvGmTZuc/fbaq/C23j+2fm2/TG937NjRxpMnT45bvzJlyjjbO3emdjSJMUb2vFfRpfqcUuLScU55PjMnHz+jAwcOtHG5cuVsvHz5cmc//X1XpUoVG8+YMcPZT5dVqlTJKdu1a5eNv/rqKxt//PHHRa12ysQ7p7yiJCIiCsiagdtHHXWUs33ZZZfZ+P3337fxsGHDnP30XyUhiQ6D6dOnj40XLnQXNLj99ttt7F9B6ivbROtERJRJfmasWbNmNtbffxUqVHD2W7t2rY07depkY53985+3Y4c7r7n+Dq1bN7sXa+IVJRERUQAbSiIiogA2lERERAHBKezSfffV3nvvXsnK7/PT/Xz77rtvoc8BgA0bNqS0Th062EW2MXfuXKesS5cuNh4+fLhTpnP9qbgDNh/vqCvteNerq2LFis72ww8/bONevXo5ZWeddZaN9T0LmZQPn9HTTz/d2T733HNtPGfOHBuXL++uq71t2zYbx+vXBNzvcf0cAChbdvctMvqOWH3nrV+PdONdr0RERElgQ0lERBSQ9uEhoZRkixYtbFy5srsM2THHHGPjI4880sbNmzd39tMDWnUqx7+deebMmTZu27atUzZx4kQbb9y40cZ6OAgAtG/f3saffvqpU7Z9+3YQUVj16tVt3K+fu1raFVdcYWN/mMHZZ59t42xJveYD/f0JuN9j+vt0/fr1zn4NGjSwsZ4Exv8e1N/5odSrntxAtwtAyaZe4+EVJRERUQAbSiIiogA2lERERAFp76MMTeemp4vTtyUD7rCPlStX2tjPgS9YsMDGOh+uc+iAO+WS37/YpEkTG9esWdPG/hCQNWvW2PjNN990yubNmwcqWVdffbWN/VvK9STO2T49VmkyePBgG5922mlO2YQJE2ys+ysBYPHixWmtV2lVo0YNZ3vz5s023rp1q4396eeWLFli4/r169vYH76nt/0+St23qb9b4y10kUnZVyMiIqIswoaSiIgoIO2p19DMP/o24EWLFjllOnWmZ4XQqVEAGDFihI31WpL+7cxt2rQp9DkAsGLFCht369bNxn66oU6dOjb2b6umkjdy5EgbT5o0ySnzU++UOfpc6JUmfPfcc4+Np06dms4qUcw+++zjbMcbzlG1alVnP52i1d/x/pqT/neopofi+esMZxteURIREQWw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APrhhx9u43vvvdfZ75577rFxLiyGzeEhv7V06VJn2x9SFI9erSjRWYDSgcNDiIiIksCGkoiIKCCjw0NCC34eeuihNh4zZoyN9RAGwJ1NRQ8R0KlWADj55JNtPG3atKJXNk/olJafJvS3k+FPUq3pidCToSdVp9zRoUMHG+t0tp61BciNdCv91pQpU2zsz8alu/b08DAAOP30023sD9nLNryiJCIiCmBDSUREFMCGkoiIKCCjfZR6kWS/31CvOKAXA23ZsmXc4+l+ye7duztlego2IkqfI444wtnWw31uvvlmG+u+Lcot//znP22sv6v9oV6631kv4gzk1r0ivKIkIiIKYENJREQUkDXDQ/zZ4/WsPWeeeaaNmzZt6uw3atQoG/fp08fGTLUSZcYZZ5zhbOshQ8OGDSvh2lA6HHLIITbWQ0D8IT7fffedjadOnZr+iqUJryiJiIgC2FASEREFZHRSdIqPEy7nn3Sc02w5n3rya3+RgaFDh9rYnwg9l5Wmz2izZs2cbT1bWv369W28efNmZ7+uXbvaeOLEiempXApxUnQiIqIksKEkIiIKYENJREQUkNHhIUSUH1auXGljPXSA8sO8efOcbb3ah+6j1AumA0C7du1snAt9lPHwipKIiCiADSUREVEAU69ERJQSy5cvd7bffvvtDNUktXhFSUREFMCGkoiIKIANJRERUQCnsMtSpWl6rNIin6ewK434Gc0/nMKOiIgoCWwoiYiIAoKpVyIiotKOV5REREQBbCiJiIgC2FASEREFsKEkIiIKYENJREQUwIaSiIgo4P8B2RdNRBNWY7EAAAAASUVORK5CYII=">
#     <br>
#   </p> 
# </td>
# </tr></table>

# ## Results discussion and Inference: Part-2
# 
# The above image shows the performance of 4-neuron-model with binary crossentropy loss (with 16 classes) as compared to the baseline 16-neuron-model with categorical crossentropy.
# 
# <span style= "color:#E83E8C;">2. What is the reason for increased accuracy of the 4-neuron-model for 16 classes compared to 10 classes?</span>
# 
# The answer is in the binary encoding of image labels. If you take a closer look at the binary encodings of numbers from $0$ to $9$, we see that $9$ is encoded as $1001$. This means the first digit ($1$) is not fired for all the other $9$ classes and is not really used during the training process.
# 
# On the flip side, when $16$ classes come into picture, all the $4$ neurons are used during training and have a better training/validation accuracy.

# ## Results discussion and Inference: Part-3
# 
# As a conclusion to this article lets take a look at instances where the 4-neuron-model and the baseline model have different predictions for the same image. Below is a picture showing a small sub-set of such instances.
# 
# _P.S. A gentle reminder that you can run the code in this article easily by using the rocket <i class="fa fa-rocket" aria-hidden="true"></i> icon on the top of the page._
# 
# <span style= "color:#E83E8C;">$4$ indicates the 4-neuron-model's prediction, b indicates the baseline model's prediction, A is the actual label of the instance under consideration.</span>

# In[ ]:


test_array_short = test_array_x[:1000]
y_test_short = test_array_y[:1000]

model_predictions = model.predict(test_array_short)
baseline_predictions = baseline.predict(test_array_short)

model_predictions = np.array([final_dict[decode(np.round(i))] for i in model_predictions])
baseline_predictions = np.array([final_dict[np.argmax(i)] for i in baseline_predictions])

print(len(model_predictions))
print(len(baseline_predictions))

bool_array = model_predictions==baseline_predictions
final = np.argwhere(bool_array==False)


# In[ ]:


fig2 = plt.figure(figsize=(12,12))

columns2 = 3
rows2 = 3
axes2 = []

print("\n")

# printing 16 training images
np.random.seed(10)
for i in range(1, columns2*rows2 +1):
    idx = random.sample(list(final),1)[0][0]
    img = test_array_short[idx]
    img1 = img.reshape(28,28)
    img_tensor = np.expand_dims(img, axis=0)
    axes.append(fig2.add_subplot(rows, columns, i))
    subplot_title = "4 -> {}, b -> {}, A -> {}".format(model_predictions[idx], baseline_predictions[idx], final_dict[test_array_y[idx]])
    axes[-1].set_title(subplot_title)
    plt.imshow(img1, interpolation='nearest', cmap=plt.get_cmap('gray'))
    plt.axis('off')
    
plt.show()


# <p style="text-align:center"><img src="images/index_33_1.png"/></p>
