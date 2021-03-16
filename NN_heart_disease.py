import numpy as np
import pandas as pd

'''We will use ANN to predict if the patient is prone to heart disease or not given a set of input parameters like age,sex, blood pressure etc. 
These will be our input parameters for our input layer. We will be using a 2 Layer ANN with ReLU and Sigmoid activation functions. This ANN is 
parametrized by specific values of learning rate (alpha), number of iterations and number of nodes for the hidden layer. You can use different 
values for these parameters and see if you get a better accuracy.'''

#Perform standard scaling on the data set
def StandardScaler(Data):
    for feature in range (0,Data.shape[1]-1):
        Current_feature = Data.iloc[:,feature]
        feature_std = (Current_feature - np.mean(Data.iloc[:,feature]))/(np.std(Data.iloc[:,feature]))
        Data.iloc[:,feature].update(feature_std)
    return Data
    
#Name the headers
headers =  ['age', 'sex','chest_pain','resting_blood_pressure','serum_cholestoral','fasting_blood_sugar','resting_ecg_results','max_heart_rate_achieved','exercise_induced_angina','oldpeak',"slope of the peak",'num_of_major_vessels','thal', 'heart_disease']
#Read the data file
df = pd.read_csv('/Users/nikhil/Data/ML_examples/heart.dat',sep=' ',names=headers)
df.isna().sum()#check for NaN
df.dtypes#check for data types. Encode to numeric if catagorical variables are involved
df = StandardScaler(df)
#Replace the heart-disease value of 1,2 to 1,0
df['heart_disease']=df['heart_disease'].replace(to_replace=[1,2], value=[0, 1])

#Shuffle data set
shuffle_df = df.sample(frac=1)
train_size = int(0.7 * len(df))
#Split it in train and test set
train_df = shuffle_df[:train_size]
test_df = shuffle_df[train_size:]

#Split into target and independent variables
y_train = np.array(train_df['heart_disease'])
y_test = np.array(test_df['heart_disease'])
x_train = np.array(train_df.drop(columns=['heart_disease']))
x_test =np.array(test_df.drop(columns=['heart_disease']))
#Add ones as the last column in x_train and x_test to take care of the intercept. This way you dont have to update b values separately.
x_train = np.c_[x_train,np.ones(np.shape(x_train)[0])]
x_test = np.c_[x_test,np.ones(np.shape(x_test)[0])]

#Set Hidden Layers and nodes for NN
Hidden_Layers = 1
Nodes_Hidden_Layers = 10
N_Layers = 1 + Hidden_Layers + 1 #Input + Hidden + Output
Layers = np.zeros([N_Layers])
Layers[0] = np.shape(x_train)[1]
Layers[1] = Nodes_Hidden_Layers
Layers[2] = 1

#Initialize weights
#np.random.seed(1)
W1 = np.random.randn(int(Layers[0]),int(Layers[1]))
W2 = np.random.randn(int(Layers[1]),int(Layers[2]))

#Define activation functions
def relu(x):
    x[x<0] = 0
    return x
    
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
    
def loss(y,y_pred):
    return 0.5*((y_pred - y)**2)
    
def dRelu(x):
    x[x<0] = 0
    x[x>0] = 1
    return x


#Initialize Cost function array. Set number of iterations and learning rate (alpha)
Loss_function = []
epochs = 500
alpha = 0.0001
for i in range (0,epochs):
    #Forward propagation
    Z1 = np.dot(x_train,W1)
    A1 = relu(Z1)
    Z2 = np.dot(A1,W2)
    y_pred = sigmoid(Z2)[:,0]
    Loss = loss(y_train,y_pred)
    Loss_function.append(Loss.sum())
    #Backward porpagation
    dloss_dW2 = np.dot((y_pred - y_train)*d_sigmoid(Z2)[:,0],A1)
    dloss_dW1 = np.dot((np.dot(((y_pred - y_train)*d_sigmoid(Z2)).T,dRelu(Z1))).T,x_train)
    dloss_dW2 = dloss_dW2.reshape(1,-1)
    #Update weights
    W1 -= (alpha*dloss_dW1.T)
    W2 -= (alpha*dloss_dW2.T)
    

def predict(X,W1,W2):
    Z1 = np.dot(X,W1)
    A1 = relu(Z1)
    Z2 = np.dot(A1,W2)
    y_pred = sigmoid(Z2)
    return np.round(y_pred)


def acc(y, yhat):
    acc = (sum(y == yhat) / len(y) * 100)
    return acc
    
y_pred_test = predict(x_test,W1,W2)[:,0]
y_pred_train = predict(x_train,W1,W2)[:,0]

test_ac = acc(y_test,y_pred_test)
train_ac = acc(y_train,y_pred_train)

print("Train accuracy is " + str(train_ac))
print("Test accuracy is " + str(test_ac))

