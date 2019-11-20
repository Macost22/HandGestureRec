"""
Created on Thu Oct 24 15:06:36 2019

+++++++.I+.@author: Melissa Acosta
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import normalize

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivative_sigmoid(x):
    return x * (1 - x)

def softmax(x):
    expA = np.exp(x)
    return expA / expA.sum(axis=1, keepdims=True)

#Iris dataset
np.random.seed(42)
# Load dataset.
iris = load_iris()
inputs, output = iris.data, iris.target
#split data into training and test data.
X1, test_X, train_y, test_y = train_test_split(inputs, output, train_size=0.8, test_size=0.2, random_state=123)

##train data
y1 = np.reshape(train_y, (120,1))
y= np.zeros((120, 3))
for i in range(120):
    y[i, y1[i]] = 1
    
X=normalize(X1, axis=0)

## test data
Xt=normalize(test_X, axis=0)
aXt = np.zeros((len(X), 4),float)
aXt[0:len(Xt),0:4] = Xt
y1t= np.reshape(test_y, (30,1))
yt=np.zeros((30,3))
for l in range(30):
    yt[l, y1t[l]] = 1

#parametros
epoch = 10000 # number of training iterations
learning_rate = 0.01

# Dimension of each layer s
d_in = X.shape[1] # number of features in the input dataset
d_h1 = 4  # neurons hidden layers + 1
d_out = 3 # output layer
# number of hidden layers
nhlayers1= 5
accuracy = np.zeros((10, 10),float)
for nhlayers in range(nhlayers1):
    for d_h in range(1,d_h1):
        print("neuronas", d_h, "layer", nhlayers)
        total_correct = 0
        if nhlayers < 3:
             # 1 hidden layer
            if nhlayers == 0:   
                wh = np.random.uniform(size=(d_in, d_h))
                bh = np.random.uniform(size=(1, d_h))
                wout = np.random.uniform(size=(d_h, d_out))
                bout = np.random.uniform(size=(1, d_out))
                for i in range(epoch):
                    # Forward pass
                    h = sigmoid(X.dot(wh) + bh)
                    y_pred = softmax(h.dot(wout) + bout)
                    
                    # Compute and print loss
                    #loss = (y_pred - y).sum()
                    sum_score = 0.0
                    for t in range(len(y)):
                        for l in range(len(y[t])):
                            sum_score += y[t][l]*np.log(1e-15 + y_pred[t][l])
                    mean_sum_score = 1.0/ len(y)*sum_score
	                    
                       
                    
	
                    #loss = np.sum(-y * np.log(y_pred))
                    if i % 1 == 0:
                        print('Epoch', i, ':', -mean_sum_score)
                        
                        # Backpropagation to compute gradients
                    grad_y_pred = (y - y_pred) #* derivative_sigmoid(y_pred)
                    grad_wout = h.T.dot(grad_y_pred)
                    grad_bout = np.sum(grad_y_pred, axis=0, keepdims=True)
                        
                    grad_h = grad_y_pred.dot(wout.T) * derivative_sigmoid(h)
                    grad_wh = X.T.dot(grad_h)
                    grad_bh = np.sum(grad_h, axis=0, keepdims=True)
                        
                        # Update weights and biases
                    wout += grad_wout * learning_rate
                    bout += grad_bout * learning_rate
                    wh += grad_wh * learning_rate
                    bh += grad_bh * learning_rate
            #test
                h = sigmoid(Xt.dot(wh) + bh)
                y_predt = softmax(h.dot(wout) + bout)
                for n in range(len(Xt)):
                    if (np.argmax(yt, axis=1)[n] ==  np.argmax(y_predt, axis=1)[n]):
                        total_correct += 1
                        accuracy[nhlayers,d_h-1] = total_correct/len(Xt)*100
               
             # 2 hidden layer            
            elif nhlayers == 1:
                wh = np.random.uniform(size=(d_in, d_h))
                bh = np.random.uniform(size=(1, d_h))
                
                wh1 = np.random.uniform(size=(d_h, d_h))
                bh1 = np.random.uniform(size=(1, d_h))
                
                wout = np.random.uniform(size=(d_h, d_out))
                bout = np.random.uniform(size=(1, d_out))
                for i in range(epoch):
                    # Forward pass
                    h = sigmoid(X.dot(wh) + bh)
                    h1 = sigmoid(h.dot(wh1)+ bh1)
                    y_pred = softmax(h1.dot(wout) + bout)
                    
                    # Compute and print loss
                    #loss = (y_pred - y).sum()
                    
                    sum_score = 0.0
                    for t in range(len(y)):
                        for l in range(len(y[t])):
                            sum_score += y[t][l]*np.log(1e-15 + y_pred[t][l])
                    mean_sum_score = 1.0/ len(y)*sum_score
	                    
                    #loss = np.sum(-y * np.log(y_pred))
                    if i % 1 == 0:
                       print('Epoch', i, ':', -mean_sum_score)
                        
                    # Backpropagation to compute gradients
                    grad_y_pred = (y - y_pred) #* derivative_sigmoid(y_pred)
                    grad_wout = h1.T.dot(grad_y_pred)
                    grad_bout = np.sum(grad_y_pred, axis=0, keepdims=True)
                        
                    grad_h1 = grad_y_pred.dot(wout.T) * derivative_sigmoid(h1)
                    grad_wh1 = h.T.dot(grad_h1)
                    grad_bh1 = np.sum(grad_h1, axis=0, keepdims=True)
                        
                    grad_h = grad_h1.dot(wh1.T) * derivative_sigmoid(h)
                    grad_wh = X.T.dot(grad_h)
                    grad_bh = np.sum(grad_h, axis=0, keepdims=True)
                        
                        # Update weights and biases
                    wout += grad_wout * learning_rate
                    bout += grad_bout * learning_rate
                        
                    wh1 += grad_wh1 * learning_rate
                    bh1 += grad_bh1 * learning_rate
                        
                    wh += grad_wh * learning_rate
                    bh += grad_bh * learning_rate
                #test    
                h = sigmoid(Xt.dot(wh) + bh)
                h1 = sigmoid(h.dot(wh1)+ bh1)
                y_predt = softmax(h1.dot(wout) + bout)
                for n in range(len(Xt)):
                    if (np.argmax(yt, axis=1)[n] ==  np.argmax(y_predt, axis=1)[n]):
                        total_correct += 1
                        accuracy[nhlayers,d_h-1] = total_correct/len(Xt)*100
             # 3 hidden layer
            else:
                wh = np.random.uniform(size=(d_in, d_h))
                bh = np.random.uniform(size=(1, d_h))
                
                wh1 = np.random.uniform(size=(d_h, d_h))
                bh1 = np.random.uniform(size=(1, d_h))
                
                wh2 = np.random.uniform(size=(d_h, d_h))
                bh2 = np.random.uniform(size=(1, d_h))
                
                wout = np.random.uniform(size=(d_h, d_out))
                bout = np.random.uniform(size=(1, d_out))
                
                for i in range(epoch):
                    # Forward pass
                    h = sigmoid(X.dot(wh) + bh)
                    h1 = sigmoid(h.dot(wh1) + bh1)
                    h2 = sigmoid(h1.dot(wh2)+ bh2)
                    y_pred = softmax(h2.dot(wout) + bout)
                    
                    # Compute and print loss
                    #loss = (y_pred - y).sum()
                    
                    sum_score = 0.0
                    for t in range(len(y)):
                        for l in range(len(y[t])):
                            sum_score += y[t][l]*np.log(1e-15 + y_pred[t][l])
                    mean_sum_score = 1.0/ len(y)*sum_score
	                    
                    loss = np.sum(-y * np.log(y_pred))
                    if i % 1 == 0:
                        print('Epoch', i, ':', -mean_sum_score)
                        
                    # Backpropagation to compute gradients
                    grad_y_pred = (y - y_pred) #* derivative_sigmoid(y_pred)
                    grad_wout = h2.T.dot(grad_y_pred)
                    grad_bout = np.sum(grad_y_pred, axis=0, keepdims=True)
                        
                    grad_h2 = grad_y_pred.dot(wout.T) * derivative_sigmoid(h2)
                    grad_wh2 = h1.T.dot(grad_h2)
                    grad_bh2 = np.sum(grad_h2, axis=0, keepdims=True)
                        
                    grad_h1 = grad_h2.dot(wh2.T) * derivative_sigmoid(h1)
                    grad_wh1 = h.T.dot(grad_h1)
                    grad_bh1 = np.sum(grad_h1, axis=0, keepdims=True)
                        
                    grad_h = grad_h1.dot(wh1.T) * derivative_sigmoid(h)
                    grad_wh = X.T.dot(grad_h)
                    grad_bh = np.sum(grad_h, axis=0, keepdims=True)
                        
                    # Update weights and biases
                    wout += grad_wout * learning_rate
                    bout += grad_bout * learning_rate
                    wh2 += grad_wh2 * learning_rate
                    bh2 += grad_bh2 * learning_rate  
                    wh1 += grad_wh1 * learning_rate
                    bh1 += grad_bh1 * learning_rate  
                    wh += grad_wh * learning_rate
                    bh += grad_bh * learning_rate 
                #test
                h = sigmoid(Xt.dot(wh) + bh)
                h1 = sigmoid(h.dot(wh1) + bh1)
                h2 = sigmoid(h1.dot(wh2)+ bh2)
                y_predt = softmax(h2.dot(wout) + bout)
                for n in range(len(Xt)):
                    if (np.argmax(yt, axis=1)[n] ==  np.argmax(y_predt, axis=1)[n]):
                        total_correct += 1
                        accuracy[nhlayers,d_h-1] = total_correct/len(Xt)*100
        
        # From 4 hidden layer to nhlayer+1         
        else:
        ### Neural network hidden layer_1,.., layer_nhlayers+1 ==> from 4 hidden layers
        ###LAYERS
        
        # Matrix zeros for hidden layers
            hweightmatrix = np.zeros((d_h,d_h,nhlayers),float)
            hbiasmatrix = np.zeros((1,d_h,nhlayers),float)
        
        # Weight and bias initialization hidden layer_1
            wh = np.random.uniform(size=(d_in, d_h))
            bh = np.random.uniform(size=(1, d_h))
        
            #Weight and bias initialization hidden layer_2, layer_3, ...., layer_nhleyrs 
            for i in range(nhlayers):
                hweightmatrix[:,:,i] = np.random.uniform(size=(d_h, d_h))
                hbiasmatrix[:,:,i] = np.random.uniform(size=(1, d_h))
        
            #Weight and bias initialization output layer
            wout = np.random.uniform(size=(d_h, d_out))
            bout = np.random.uniform(size=(1, d_out))
            
            # Training hActivationMatriz = Output layer Matriz, hgradmatrix = gradient of the local fiel  Matrix, 
            # hgradweightmatrix = hgradmatrix * input layer, hgradweightmatrix = bias matrix
            hActivationMatrix = np.zeros((len(X),d_h,nhlayers),float)
            hgradmatrix = np.zeros((len(X),d_h,nhlayers),float)
            hgradweightmatrix = np.zeros((d_h,d_h,nhlayers),float)
            hgradbiasmatrix = np.zeros((1,d_h,nhlayers),float)
            
            ##Train
            
            for i in range(epoch):
            
                ## Forward pass
                
                # Hidden layer_1 output
                h = sigmoid(X.dot(wh) + bh) # First layer activation or h
            
                # Hidden layer_2 output
                hActivationMatrix[:,:,0] = sigmoid(h.dot(hweightmatrix[:,:,0]) + hbiasmatrix[:,:,0])
            
                # Hidden layer_3,..., Layer_nhlayers outputs
                for j in range(1,nhlayers): 
                    hActivationMatrix[:,:,j] = sigmoid(hActivationMatrix[:,:,j-1].dot(hweightmatrix[:,:,j]) + hbiasmatrix[:,:,j])
                
                # Last layer output or y_pred
                y_pred = softmax(hActivationMatrix[:,:,-1].dot(wout) + bout)
            
                # Compute and print loss
                sum_score = 0.0
                for t in range(len(y)):
                    for l in range(len(y[t])):
                        sum_score += y[t][l]*np.log(1e-15 + y_pred[t][l])
                mean_sum_score = 1.0/ len(y)*sum_score
	                    
                #
                #loss1 = (y_pred - y).sum()
                #loss2 = np.sum(-y * np.log(y_pred))
                if i % 1 == 0:
                    print('Epoch', i, ':', -mean_sum_score)
                   # print('Epoch2', i, ':', loss2)
                
        
                ## Backpropagation to compute gradients
            
                # Output layer
                grad_y_pred = (y - y_pred) #* derivative_sigmoid(y_pred) # Local gradient
                grad_wout = hActivationMatrix[:,:,-1].T.dot(grad_y_pred) # Local gradiente * input to the layer
                grad_bout = np.sum(grad_y_pred, axis=0, keepdims=True)   # Gradient bias
                
                # Local gradient Hidden layer_nhlayer
                hgradmatrix[:,:,0] = grad_y_pred.dot(wout.T) * derivative_sigmoid(hActivationMatrix[:,:,-1])
                hgradweightmatrix[:,:,0] = hActivationMatrix[:,:,-2].T.dot(hgradmatrix[:,:,0])
                hgradbiasmatrix[:,:,0] = np.sum(hgradmatrix[:,:,0], axis=0, keepdims=True)
                
                # Local gradient hidden layer_nhlayer-1,..., layer_3
                for j in range(1,nhlayers-1):
                    hgradmatrix[:,:,j] = hgradmatrix[:,:,j-1].dot(hweightmatrix[:,:,-j].T) * derivative_sigmoid(hActivationMatrix[:,:,-j-1])
                    hgradweightmatrix[:,:,j] = hActivationMatrix[:,:,-j-2].T.dot(hgradmatrix[:,:,j])
                    hgradbiasmatrix[:,:,j] = np.sum(hgradmatrix[:,:,j], axis=0, keepdims=True)
                    
                    # Local gradient hidden layer_2
                    hgradmatrix[:,:,-1] = hgradmatrix[:,:,-2].dot(hweightmatrix[:,:,-2].T) * derivative_sigmoid(hActivationMatrix[:,:,0])
                    hgradweightmatrix[:,:,-1] = h.T.dot(hgradmatrix[:,:,-1])
                    hgradbiasmatrix[:,:,-1] = np.sum(hgradmatrix[:,:,-1], axis=0, keepdims=True)
                    
                    # Local gradient hidden layer_1
                    grad_h = hgradmatrix[:,:,-1].dot(hweightmatrix[:,:,0].T) * derivative_sigmoid(h)
                    grad_wh = X.T.dot(grad_h)
                    grad_bh = np.sum(grad_h, axis=0, keepdims=True)
                    
                    
                    ## Update weights and biases
                    
                    # Output layer
                    wout += grad_wout * learning_rate
                    bout += grad_bout * learning_rate
                    
                    # Hidden layer_2, ... , layer_nhlayer
                    for j in range(nhlayers):
                        hweightmatrix[:,:,-j-1] +=  hgradweightmatrix[:,:,j] * learning_rate
                        hbiasmatrix[:,:,-j-1] +=  hgradbiasmatrix[:,:,j] * learning_rate
            
                    # Hidden layer_1
                    wh += grad_wh * learning_rate
                    bh += grad_bh * learning_rate
            #test
            h = sigmoid(aXt.dot(wh) + bh) # First layer activation or h
            hActivationMatrix[:,:,0] = sigmoid(h.dot(hweightmatrix[:,:,0]) + hbiasmatrix[:,:,0])
            for j in range(1,nhlayers): 
                hActivationMatrix[:,:,j] = sigmoid(hActivationMatrix[:,:,j-1].dot(hweightmatrix[:,:,j]) + hbiasmatrix[:,:,j])
                y_predt = softmax(hActivationMatrix[:,:,-1].dot(wout) + bout)
            for n in range(len(Xt)):
                    if (np.argmax(yt, axis=1)[n] ==  np.argmax(y_predt, axis=1)[n]):
                        total_correct += 1
                        accuracy[nhlayers,d_h-1] = total_correct/len(Xt)*100

                
        