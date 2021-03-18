import numpy as np
import pandas as pd
from keras.datasets import mnist

'''We will train a CNN to recognize the hand written digits from the MNIST data set.
The layers will be as follows:

Input => Convolution => Sigmoid => Linear => Softmax => Output

We are using a 4 by 4 convolution filter with depth 3 but you can experiment with different depths and
filter sizes and test the accuracy. For this example we have only selected the first 200 images. For
convolution process, we have used stride=1 and pad=0. The ideal way to do this is to use im2col algorithm
where you can set the stride and padding values. We will use get_patches function for stride=1/pad=0'''

#define get_patches function
def get_patches(input,filt):
    '''This function performs the get_patches algorithm. This function selects patches from the input image to which
       can be used forthe convolution procedure.
    
    
    Input:
    
        input: Input image array. Must be a (k,m,n) array with k images each of 2-D shape (m,n).
    
        filt: Convolution filter. Must be a (d,x,y) array with depth d and a 2-D filter of shape (x,y).
        
    Output:
        
        New_iamge: Output array of shape (k,(m-x+1)*(n-y+1),x*y)'''
    
    output_X,output_Y = int((np.shape(input)[1] - filt.shape[1]) + 1),int((np.shape(input)[2] - filt.shape[2]) + 1)
    N_batches = int(output_X*output_Y)
    New_image = []
    k=0
    for k in range (0,np.shape(input)[0]):
        i=0
        for i in range (0,output_X):
            j=0
            for j in range(0,output_Y):
                    New_image.append(input[k,:,:][i:i+filt.shape[1],j:j+filt.shape[2]])
    New_image = np.reshape(New_image,(input.shape[0],N_batches,int(filt.shape[1]*filt.shape[2])))
    return New_image

#define activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
    
def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def loss(y,yhat):
    return np.sum(-y*np.log(yhat))


# loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#selecting the first 200 images
x_train = x_train[:200]
y_train = y_train[:200]
x_test = x_test[:200]
y_test = y_test[:200]

#Normalizing x
x_train = x_train/np.max(x_train)
x_test = x_test/np.max(x_test)

#One_hot_encoding for training set output
one_hot_labels_train = np.zeros((len(y_train),10))
for num in range (0,len(y_train)):
    one_hot_labels_train[num,y_train[num]] = 1
    
#One_hot_encoding for test set output
one_hot_labels_test = np.zeros((len(y_test),10))
for num in range (0,len(y_test)):
    one_hot_labels_test[num,y_test[num]] = 1
    
#update test train outputs
y_train = one_hot_labels_train
y_test = one_hot_labels_test

#initialize convolution filter size and output convolved image size
f = np.random.randn(3,4,4)
input_image = get_patches(x_train,f)
OX,OY =int(((x_train.shape[1] - f.shape[1])) + 1 ),int(((x_train.shape[2] - f.shape[2])) + 1)
N_batches = OX*OY

#initialize the weights for the linear layer
W = np.random.randn(int(OX*OY*f.shape[0]),10)



#set the number of iteration, learning rate (alpha) for SGD method
epochs = 300
alpha = 0.015
Loss_function = []
for epochs in range (0,epochs):
    #Forward propagation
    filter_image = np.reshape(f,(f.shape[0],int(f.shape[1]*f.shape[2])))
    Conv_image = np.dot(input_image,filter_image.T) #Convolution operation
    Z1 = Conv_image
    A1 = sigmoid(Z1)
    A1 = np.reshape(A1,(x_train.shape[0],int(N_batches*f.shape[0])))
    Z2 = np.dot(A1,W)
    y_pred = softmax(Z2)
    Loss = loss(y_train,y_pred)
    Loss_function.append(Loss)
    #Backward propagation
    dJ_dW = np.dot(A1.T,(y_pred-y_train))
    dZ1_df = input_image
    dA1_dZ1 = d_sigmoid(Z1)
    dJ_dZ2 = (y_pred-y_train)
    dZ2_dA1 = W
    dJ_dA1 = np.dot(dJ_dZ2,dZ2_dA1.T)
    dJ_dZ1 = np.reshape(dJ_dA1,(x_train.shape[0],N_batches,f.shape[0]))*dA1_dZ1
    dZ1_df = np.reshape(dZ1_df,(int(x_train.shape[0]*N_batches),int(f.shape[1]*f.shape[2])))
    dJ_dZ1 = np.reshape(dJ_dZ1,(int(x_train.shape[0]*N_batches),f.shape[0]))
    dJ_df = (np.dot(dZ1_df.T,dJ_dZ1)).T
    dJ_df = np.reshape(dJ_df,(f.shape[0],f.shape[1],f.shape[2]))
    #update weights
    f-= alpha*dJ_df
    W-= alpha*dJ_dW

#Define prediction function
def predict(X,f,W):
    f = np.reshape(f,(f.shape[0],int(f.shape[1]*f.shape[2])))
    Z1 = np.dot(X,f.T)
    A1 = sigmoid(Z1)
    A1 = np.reshape(A1,(X.shape[0],int(N_batches*f.shape[0])))
    Z2 = np.dot(A1,W)
    y_pred = softmax(Z2)
    y_new = np.zeros(np.shape(y_pred))
    #Convert softmax output to 0s and 1s
    for num in range(len(y_pred)):
        y_new[num,np.argmax(y_pred[num,:])] = 1
    return y_new



#predict test and train accuracy
test_images = get_patches(x_test,f)#need to get the test images into convolution array format
y_pred_test = predict(test_images,f,W)
y_pred_train = predict(input_image,f,W)

#calculate accuracy
def acc(y, yhat):
    acc = (sum(y == yhat) / len(y) * 100)
    return acc
    
test_ac = np.mean(acc(y_test,y_pred_test))
train_ac = np.mean(acc(y_train,y_pred_train))

print("Train accuracy is " + str(train_ac))
print("Test accuracy is " + str(test_ac))




