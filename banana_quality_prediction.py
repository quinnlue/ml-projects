### IMPORTANT DISCLAIMER ###
# This project was written following Andrew Ng's course on machine learning

# While I wrote and fully understand all of the code below,
# The ideas are not original
# This code will not pass any plagiarism checker 
# It is not intended to

# Also, this project does not follow many principles of a high quality model (i.e. testing on the training set)

# This project's focus is understand the math behind gradient descent
# Computing partial derivatives by hand in back prop 
# Performing gradient descent with just numpy




import numpy as np
import pandas as pd

df = pd.read_csv('C:\\data_sets\\banana_quality.csv')


data = df.loc[: , ['Size' , 'Weight' , 'Sweetness' , 'Softness' , 'HarvestTime' , 'Ripeness' , 'Acidity']]


ban_data = data.to_numpy()
Y_data = df.loc[: , ['Quality']]
Y = Y_data.to_numpy().T
Y = np.where(Y == 'Good' , 1 , 0)
X = ban_data.T
m = ban_data.shape[0]



def init_params(X , Y):
    h = 1
    W1 = np.random.randn(X.shape[0] , h)
    b1 = np.zeros((h , 1))
    W2 = np.random.randn(h , Y.shape[0])
    b2 = np.zeros((Y.shape[0] , 1))


    params = {'W1': W1,
              'b1': b1,
              'W2': W2,
              'b2': b2}
    return params


def sigmoid(Z2):
    A2 = 1/(1+np.exp(Z2))
    return A2




def forward_prop(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1.T , X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2.T , A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1' : Z1,
             'A1' : A1,
             'Z2' : Z2,
             'A2' : A2
             }
    return A2 , cache


def cost(A2 , Y):
    logprobs = np.multiply(Y ,np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
    cost = (-1/m) * np.sum(logprobs)
    return cost





def back_prop(params , cache , X , Y):
    m = X.shape[1]

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2 , A1.T).T

    db2 = 1/m * np.sum(dZ2, axis = 1 , keepdims= True)
    dZ1 = np.dot(W2 , dZ2) * (1 - np.power(A1 , 2))
    dW1 = 1/m * np.dot(dZ1 , X.T).T

    db1 = 1/m * (np.sum(dZ1 , axis= 1 , keepdims=True))


    grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}

    return grads



def update_params(params , grads , learning_rate):
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = params['W1'] + (dW1 * learning_rate)
    b1 = params['b1'] + (db1 * learning_rate)
    W2 = params['W2'] + (dW2 * learning_rate)
    b2 = params['b2'] + (db2 * learning_rate)

    
    params = {'W1': W1,
              'b1': b1,
              'W2': W2,
              'b2': b2}

    return params




def accuracy(A2 , Y):
    predictions = np.rint(A2)
    accuracy = np.mean(predictions == Y)
    return accuracy




def nn_model(epochs):
    params = init_params(X, Y)

    for i in range(epochs):
        
 
        A2, cache = forward_prop(X , params)

        grads = back_prop(params , cache , X , Y)

        params = update_params(params , grads , learning_rate = 0.05)

        if i %1000 == 0:
            print(f'Epoch: {i} -- Accuracy {accuracy(A2,Y)*100}%')

    print(f'Epoch: {i} -- Accuracy {(accuracy(A2,Y)*100)}%')

nn_model(10000)