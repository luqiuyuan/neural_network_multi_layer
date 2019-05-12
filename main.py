import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

NUM_HIDDEN_LAYERS = 4
DIMENSIONS_LAYERS = [49152, 200, 200, 200, 200, 1] # Dimensions of input layer and hidden layers
NUM_ITERATIONS = 2
m = 20 # Number of samples
alpha = 0.01 # Learning rate
SIZE_TESTING = 10 # Size of testing dataset

# Read inputs
def readInputs(path):
    X = np.zeros((DIMENSIONS_LAYERS[0], 0))
    for filename in glob.glob(path):
        img = mpimg.imread(filename)
        X = np.hstack((X, img.reshape((DIMENSIONS_LAYERS[0], 1))))
    # Normalize X so that it is in range [0, 1]
    X = X / 256 - 0.5
    return X

# Read ground truth
def readGroundTruth(path):
    Y = np.zeros((1, 0))
    file = open(path, "r")
    for line in file:
        Y = np.hstack((Y, np.array([[int(line)]])))
    file.close()
    return Y

# Initialize parameters to small random numbers
def initializeParametersRandom():
    Ws = [None]
    Bs = [None]
    for i in range(0, NUM_HIDDEN_LAYERS+1):
        Ws.append((np.random.random((DIMENSIONS_LAYERS[i], DIMENSIONS_LAYERS[i+1])) - 0.5) * 0.01)
        Bs.append(np.zeros((1, DIMENSIONS_LAYERS[i+1])))
    return (Ws, Bs)

# Activation function sigma
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Activation function ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    y = np.copy(x)
    y[x<0] = 0
    return y

def calculateBackwardPropagation(X, Ws, Bs):
    Zs = [None]
    As = [X]

    # Calculate for hidden layers
    for i in range(1, NUM_HIDDEN_LAYERS+1):
        Z = Ws[i].T.dot(As[i-1]) + Bs[i].T
        A = relu(Z)
        Zs.append(Z)
        As.append(A)
    
    # Calculate for output layer
    Z = Ws[NUM_HIDDEN_LAYERS+1].T.dot(As[NUM_HIDDEN_LAYERS]) + Bs[NUM_HIDDEN_LAYERS+1].T
    A = sigmoid(Z)
    Zs.append(Z)
    As.append(A)

    return (Zs, As)

# Cost of a training iteration
def calculateCost(Y, A):
    return -(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / m

######################
# Training phase
######################

# Read inputs and ground truth for training phase
X = readInputs("./dataset/training/*.jpg")
Y = readGroundTruth("./dataset/training/data.txt")

# Initialize parameters
(Ws, Bs) = initializeParametersRandom()

# Training iteration
Js = np.zeros((0))
for i in range(0, NUM_ITERATIONS):
    (Zs, As) = calculateBackwardPropagation(X, Ws, Bs)
    J = calculateCost(Y, As[NUM_HIDDEN_LAYERS+1])[0, 0]
    Js = np.hstack((Js, np.array([J])))
    
