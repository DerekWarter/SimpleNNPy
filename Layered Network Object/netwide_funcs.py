import numpy as np
import math

##### ACTIVATION FUNCTIONS #####

# NUMERICALLY STABLE SOFTMAX:
# In: Array of values x in R.
# Out: Array of values y in R, yi <= 1, sum = 1.

def softmax(X):
    expSum = np.exp(X - np.max(X))
    return expSum / np.sum(expSum)

# SOFTMAX DERIVATIVE FOR OUTPUT LAYER:
# In: Array of values Y in R, Yi <= 1, sum = 1.
# Out: Diagonal axis of Jacobian matrix of values O in R.

def softmax_deriv(Y):
    #outMat = Y.reshape(-1,1)
    #return np.diagflat(outMat) - np.dot(outMat, outMat.T)
    #return np.multiply(Y,np.subtract(1,Y))
    D = []
    for i in range(len(Y)):
        for j in range(len(Y)):
            if (i == j):
                D.append(Y[i] * (1 - Y[i]))
            #else:
            #    D.append(Y[i] * (0 - Y[j]))
    return D

# SOFTMAX DERIVATIVE FOR OUTPUT LAYER:
# In: Array of values Y in R, Yi <= 1, sum = 1.
# Out: Jacobian matrix of values O in R.

def softmax_deriv_s(Y):
    outMat = Y.reshape(-1,1)
    return np.diagflat(outMat) - np.dot(outMat, outMat.T)
    #return np.multiply(Y,np.subtract(1,Y))

# RECTIFIED LINEAR UNIT (reLU):
# In: Array of values X in R
# Out: Array of values Y in R

def relu(X):
    Y = [max(0, x) for x in X]
    return Y

# RECTIFIED LINEAR UNIT DERIVATIVE:
# In: Array of values Y in R
# Out: Array of values D in R

def relu_deriv(Y):
    D = []
    for y in Y:
        if (y > 0):
            D.append(1)
        else:
            D.append(0)
    return D

# LEAKY RECITIFIED LINEAR UNIT (leaky reLU):
# In: Array of values X in R
# Out: Array of values Y in R

def lrelu(X):
    Y = []
    for x in X:
        if (x > 0):
            Y.append(x)
        else:
            Y.append(.05 * x)
    return Y

# LEAKY RECTIFIED LINEAR UNIT DERIVATIVE:
# In: Array of values Y in R
# Out: Array of values D in R

def lrelu_deriv(Y):
    D = []
    for y in Y:
        if (y > 0):
            D.append(1)
        else:
            D.append(.05)
    return D

# HYPERBOLIC TANGENT:
# In: Array of values X in R
# Out: Array of values Y in R

def tanh(X):
    return np.tanh(X)

# HYPERBOLIC TANGENT DERIVATIVE:
# In: Array of values Y in R
# Out: Array of values D in R

def tanh_deriv(Y):
    return 1 - np.tanh(Y)**2

##### LOSS FUNCTIONS #####

# CROSS ENTROPY FOR OUTPUT LAYER:
# In:
	#Y: Predicted output values for network.
	#T: Actual output values for classification sample.
# Out: Errors for softmax logits wrt target array.

def cross_entropy(Y, T):
    E = []
    for y in range(len(Y)):
        if T[y] == 1:
            E.append(-np.log(Y[y]))
        else:
            E.append(-np.log(1 - Y[y]))
    return E

# CROSS ENTROPY DERIVATIVE FOR OUTPUT LAYER:
# In:
	#Y: Predicted output values for network.
	#T: Actual output values for classification sample.
# Out: Derivative of errors for softmax logits wrt target array.

def cross_entropy_deriv(Y, T):
    return np.subtract(Y, T)

# MEAN SQUARED ERROR FOR OUTPUT LAYER:
# In: 
	#Y: Predicted output values for network.
	#T: Actual output values for classification sample.
# Out: Squared difference of Y and T divided by number of outputs.

def mean_squared_error(Y, T):
    return np.mean(np.subtract(T, Y)**2)

# SQUARED ERROR FOR OUTPUT LAYER:
# In:
	#Y: Predicted output values for network.
	#T: Actual output values for classification sample.
# Out: Squared difference of Y and T.

def squared_error(Y, T):
    return (1/2)*np.subtract(Y, T)**2

# SQUARED ERROR DERIVATIVE FOR OUTPUT LAYER:
# In:
	#Y: Predicted output values for network.
	#T: Actual output values for classification sample.
# Out: Derivative of squared difference of Y and T.

def squared_error_deriv(Y, T):
    return np.subtract(Y, T)

