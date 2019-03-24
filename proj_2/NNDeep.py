# Import Libraries
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
# A class that models the Neural Net with L-layers and
# N neurons in each layer. It also contains the functions
# for training, testing, and optimizing the Neural Network

class DeepNN:

    # Constructor to build the structure of the Neural Network
    # It accepts the layers in the format of [2,3,1] -> 2 Neuron Input Layer,
    # 3 Neuron Hidden Layer and 1 Neuron output layer
    def __init__(self, layers):
        ############################### Initialize the number of layers and neurons
        self.layers = layers
        self.num_layers = len(layers)
        self.hidden_layers = len(layers) - 2
        self.input_neurons = layers[0]
        self.output_neurons = layers[-1]

        ########## Intialize parameters for Forward Propogation
        # Initialize Weights
        self.epsilon = 0.12  # Random Weight Initialization Factor
        self.weights = []
        for i in range(self.num_layers-2):
            self.weights.append(np.random.randn(layers[i]+1, layers[i+1]+1)*2*self.epsilon - self.epsilon)
                        # We add a +1 to incorporate for weights from the +1 neuron for the bias
        self.weights.append(np.random.randn(layers[-2]+1, layers[-1])*2*self.epsilon - self.epsilon)

        self.a = [] # To keep track of activations
        self.z = [] # To keep track of layer values
        self.activations = ['sigmoid']*(self.num_layers-1) # Activations for each layer

        ######### Intialize parameters for Backward Propogation
        self.delta = []
        self.gradient = []

        # Initialize Scaling
        self.scaler = preprocessing.StandardScaler()

    ################################### Define Some Activation Functions and their derivatives ##################
    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoidPrime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def reLU(self,x):
        return np.maximum(x, 0)

    def reLUPrime(self,x):
        return np.where(x > 0, 1.0, 0.0)

    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x), axis = 0)

    def tanh(self,z):
        return np.tanh(z)

    def tanh_prime(self,x):
        return 1 - np.tanh(x)**2

    def identity(self,x):
        return x

    def identity_prime(self,x):
        return 1

    ######################################### Cost Functions #############################################
    # Least Squares
    def least_squares_cost(self,t):
        return 0.5*np.mean( (t-self.a[-1])**2 )

    # Cross Entropy Log Loss Function
    def log_loss(self,t):
        return np.mean( np.nan_to_num( -1*t*np.log(self.a[-1]) - (1-t)*np.log(1-self.a[-1]) ) )

    ######################################### Forward Feed ##############################################
    def forwardFeed(self, X, activations):
        self.activations = activations
        a = [X] # Keep Track of activations
        z = []

        # Add Bias
        c = np.ones([1,a[0].shape[0]]).reshape(a[0].shape[0],1)
        a[0] = np.concatenate((c,a[0]), axis=1)
#         print(a)
        for i in range(self.num_layers-1):
#             print(a[i])
            z.append(np.dot(a[i],self.weights[i]))
            if(activations[i] == 'sigmoid'):
                a.append(self.sigmoid(z[i]))
            elif(activations[i] == 'reLU'):
                a.append(self.reLU(z[i]))
            elif(activations[i] == 'tanh'):
                a.append(self.tanh(z[i]))
            elif(activations[i] == 'softmax'):
                a.append(self.softmax(z[i]))
            elif(activations[i] == 'identity'):
                a.append(self.identity(z[i]))
        self.a = a
        self.z = z

    def backPropogate(self,y):

        delta = []
        gradient = []
#         print('Weights:', self.weights)
        weights_flipped = self.weights[::-1]
        z_flipped = self.z[::-1]
        activations_flipped = self.a[::-1]
        activation_func_flipped = self.activations[::-1]
        delta.append(activations_flipped[0] - y)
#         print('Weights Flipped:', weights_flipped)
        for i in range(0,self.num_layers-2):
#                 print('delta: ',delta[i])
#                 print('weights_flipped: ',weights_flipped[i])
#                 print('z_flipped: ',z_flipped[i+1])
                if(activation_func_flipped[i] == 'sigmoid'):
#                     print('Sigmoid Prime')
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.sigmoidPrime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'reLU'):
#                     print('reLU Prime')
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.reLUPrime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'tanh'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.tanh_prime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'identity'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.identity_prime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'softmax'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T))

        delta = delta[::-1]

        for i in range(len(delta)):
            gradient.append( np.dot(self.a[i].T, delta[i]) )

        self.delta = delta
        self.gradient = gradient

    def learn(self, epochs, learning_rate, X, y, activations, cost_func, metrics_at=10, optimizer = '', batch_size=10, scaler_type='standard_scaler', split=False, test_size=0.25):
        start = time.time()

        if scaler_type == 'min_max_scaler':
            self.scaler = preprocessing.MinMaxScaler()
        X = self.scaler.fit_transform(X)

        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
            X = X_train
            y = y_train

        for i in range(epochs):
                if optimizer == 'sgd':
                    random_indics = np.random.randint(len(y),size=batch_size)
                    X_sgd = X[random_indics]
                    y_sgd = y[random_indics]
                    self.forwardFeed(X_sgd, activations)
                    self.backPropogate(y_sgd)
                else:
                    self.forwardFeed(X, activations)
                    self.backPropogate(y)

                for j in range(len(self.gradient)):
                    self.weights[j] = self.weights[j] - learning_rate*self.gradient[j]

                if(i%metrics_at == 0):
                    self.forwardFeed(X, activations)
                    print('Effective epoch: ', i/metrics_at + 1)
                    if(cost_func == 'log_loss'):
                        cost = self.log_loss(y)
                        print('Accuracy: ', np.mean(np.round(self.think(X))==y) * 100, '%')
                    elif(cost_func == 'least_squares'):
                        cost = self.least_squares_cost(y)
                    print('Cost: ', cost, '\n')

        if(cost_func == 'log_loss' and split):
            print('Testing Accuracy: ', np.mean(np.round(self.think(X_test))==y_test) * 100, '%')

        end = time.time()
        print('Time Taken: ', end-start, ' seconds')
        return self.weights

    def think(self,X):

        X = self.scaler.fit_transform(X)

        activations = self.activations
        a = [X] # Keep Track of activations
        z = []

        # Add Bias
        c = np.ones([1,a[0].shape[0]]).reshape(a[0].shape[0],1)
#         print(a[0].shape)
        a[0] = np.concatenate((c,a[0]), axis=1)

        for i in range(self.num_layers-1):
            z.append(np.dot(a[i],self.weights[i]))
            if(activations[i] == 'sigmoid'):
                a.append(self.sigmoid(z[i]))
            elif(activations[i] == 'reLU'):
                a.append(self.reLU(z[i]))
            elif(activations[i] == 'tanh'):
                a.append(self.tanh(z[i]))
            elif(activations[i] == 'softmax'):
                a.append(self.softmax(z[i]))
            elif(activations[i] == 'identity'):
                a.append(self.identity(z[i]))
        return a[-1]
