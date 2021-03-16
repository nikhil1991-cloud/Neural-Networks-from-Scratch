import numpy as np
import pandas as pd
import numpy as np
from keras.datasets import mnist
import random

'''We will be traing an artificial neural network to recognize the hand written digits from the mnist data set. 
Since this is an artificial neural network, we will have to reshape each 2-D image into a 1-D array. The total number 
of pixels of each image will be the variables for our input layer. If its an (m,n) image then our input variables will be m*n. 
We will be using two hidden layers with sigmoid and softmax activation functions. For this example we are using 1000 nodes 
for the 1st hidden layer. We have set the learning rate (alpha) = 0.01 and we will run SGD for 200 iterations. To see if our code 
is working correctly or not, we will only use the 1st 1000 images from the train/test dataset.'''


#Read the MNIST Data set
(x0_train, y0_train), (x0_test, y0_test) = mnist.load_data()
x0_train = x0_train[:1000,:,:]
y0_train = y0_train[:1000]
x0_test = x0_test[:1000,:,:]
y0_test = y0_test[:1000]

#Labels are 10 digits from 0-9 so we will perform one-hot encoding on the output.
one_hot_labels_train = np.zeros((len(x0_train),10))
one_hot_labels_test = np.zeros((len(x0_test),10))
for i in range(len(one_hot_labels_train)):
    one_hot_labels_train[i,y0_train[i]] = 1
for i in range(len(one_hot_labels_test)):
    one_hot_labels_test[i,y0_test[i]] = 1

#Normalize the pixel values with min max scaling
x_train = x0_train/np.max(x0_train)
x_test = x0_test/np.max(x0_test)
y_train = one_hot_labels_train
y_test = one_hot_labels_test

#Reshape the 2-D images into 1-D images
x_train = np.reshape(x_train,(x_train.shape[0],int(x_train.shape[1]*x_train.shape[1])))
x_test = np.reshape(x_test,(x_test.shape[0],int(x_test.shape[1]*x_test.shape[1])))


#Define the Layers and nodes
Hidden_Layers = 1
Nodes_Hidden_Layers = 1000
N_Layers = 1 + Hidden_Layers + 1 #Input + Hidden + Output
Layers = np.zeros([N_Layers])
Layers[0] = np.shape(x_train)[1]
Layers[1] = Nodes_Hidden_Layers
Layers[2] = 10

#Initialize the weights depending on the hidden layers
np.random.seed(42)
W1 = np.random.randn(int(Layers[0]),int(Layers[1]))
W2 = np.random.randn(int(Layers[1]),int(Layers[2]))


#define activation functions and their derivatives
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def loss(y,yhat):
    return np.sum(-y*np.log(yhat))
    
    
#set the learning rate (alpha), number of iterations and initialize cost function array
alpha = 0.01
epochs = 200
Loss_function = []
#Start the stochastic gradient descent method
for i in range (0,epochs):
    #forward propagation
    Z1 = np.dot(x_train,W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1,W2)
    y_pred = softmax(Z2)
    loss_f = loss(y_train,y_pred)
    Loss_function.append(loss_f.sum())
    #Back propagation
    dJ_dW2 = np.dot(A1.T,(y_pred-y_train))
    dZ1_dW1 = x_train
    dA1_dZ1 = d_sigmoid(Z1)
    dJ_dZ2 = (y_pred-y_train)
    dZ2_dA1 = W2
    dJ_dA1 = np.dot(dJ_dZ2,dZ2_dA1.T)
    dJ_dZ1 = dJ_dA1*dA1_dZ1
    dJ_dW1 = np.dot(dZ1_dW1.T,dJ_dZ1)
    #update weights
    W1 -= alpha*dJ_dW1
    W2 -= alpha*dJ_dW2
    
#Define the prediction function
def predict(X,W1,W2):
    Z1 = np.dot(X,W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1,W2)
    y_pred = softmax(Z2)
    y_new = np.zeros(np.shape(y_pred))
    #Convert softmax output to 0s and 1s
    for num in range(len(y_pred)):
        y_new[num,np.argmax(y_pred[num,:])] = 1
    return y_new
    
#calculate accuracy
def acc(y, yhat):
    acc = (sum(y == yhat) / len(y) * 100)
    return acc

#predict test and train accuracy
y_pred_test = predict(x_test,W1,W2)
y_pred_train = predict(x_train,W1,W2)

test_ac = np.mean(acc(y_test,y_pred_test))
train_ac = np.mean(acc(y_train,y_pred_train))


print("Train accuracy is " + str(train_ac))
print("Test accuracy is " + str(test_ac))

#You can plot the Loss_function as a function of iterations to see if it is minimized.
#You can play with different values of iterations, learning rate and hidden layer nodes to see if your accuracy improves

'''Optional: Another way to optimize this neural network is to run a Markov Chain Monte Carlo random walk in the 4-D parameter
space of iterations, learning rate, nodes and number of hidden layers for N number of steps. At each step, use the trial parameters
and train the NN on the dataset, compute the accuracy and accept the trail parameters if the accuracy is better than the previous step.
You can used Metropolis-Hasting Sampling to sample the parameter space'''
