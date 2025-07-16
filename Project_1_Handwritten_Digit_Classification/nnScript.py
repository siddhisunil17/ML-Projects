import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import itertools
import pickle
import time
import matplotlib.pyplot as plt

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))
    """
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):

  """
  # Notice that z can be a scalar, a vector or a matrix
  # return the sigmoid of input z
  """
  return 1 / (1 + np.exp(-z))

def preprocess():
    """
    Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection
    """
    mat = loadmat('"C:/Pratham/abc/WorkSpace/Sem 2/CSE 574 Intro ML/Project 1/basecode/mnist_all.mat"')  # loads the MAT object as a Dictionary

    # Initialize lists to collect data and labels
    train_data = []
    train_label = []
    validation_data = []
    validation_label = []
    test_data = []
    test_label = []

    # Spliting dataset into train validation and test
    for i in range(10):
        key = f'train{i}'
        data = mat[key]
        points = data.shape[0]
        label = np.full((points), i)
        pc = np.random.permutation(points)  
        split = int(0.9 * points)  

        train_data.append(data[pc[:split]])
        validation_data.append(data[pc[split:]])
        train_label.append(label[pc[:split]])
        validation_label.append(label[pc[split:]])

        test_point = f'test{i}'
        test_data.append(mat[test_point])
        test_label.append(np.full((mat[test_point].shape[0]), i))

    # Normalizing pixel values 
    train_data = np.vstack(train_data) / 255.0
    train_label = np.hstack(train_label)
    validation_data = np.vstack(validation_data) / 255.0
    validation_label = np.hstack(validation_label)
    test_data = np.vstack(test_data) / 255.0
    test_label = np.hstack(test_label)

    # Feature selection 
    nonzero_features = np.std(train_data, axis=0) > 0.01
    train_data = train_data[:, nonzero_features]
    validation_data = validation_data[:, nonzero_features]
    test_data = test_data[:, nonzero_features]

    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """
    % nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, the training data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 and w2 (input->hidden and hidden->output)
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer
    % training_data: matrix of training data
    % training_label: vector of truth labels
    % lambda: regularization hyper-parameter

    % Output:
    % obj_val: scalar value of error function
    % obj_grad: a single flattened vector of gradient values
    """
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    # Reshape the weight vectors into matrices
    w1 = params[:n_hidden * (n_input + 1)].reshape((n_hidden, n_input + 1))
    w2 = params[n_hidden * (n_input + 1):].reshape((n_class, n_hidden + 1))
    obj_val = 0

    # Your code here
    n = training_data.shape[0]
    input_bias = np.ones((n, 1)) 
    training_data = np.hstack((training_data, input_bias))  

    # Feedforward 
    hidden_input = np.dot(training_data, w1.T)
    hidden_output = sigmoid(hidden_input)  
    hi_bias = np.ones((hidden_input.shape[0], 1))
    hl_bias = np.hstack((hidden_output, hi_bias))  
    ol_input = np.dot(hl_bias, w2.T)
    ol_output = sigmoid(ol_input)  

    y = np.zeros((n, n_class))
    y[np.arange(n), training_label.astype(int)] = 1

    # Calculating error function using log-likelihood
    error_fun = -np.sum(y * np.log(ol_output) + (1 - y) * np.log(1 - ol_output)) / n

    # Adding regularization to error function
    regu = (lambdaval / (2 * n)) * (np.sum(w1**2) + np.sum(w2**2))
    obj_val = error_fun + regu

    # Backpropagation 
    delta_output = ol_output - y
    grad_w2 = np.dot(delta_output.T, hl_bias) / n + (lambdaval * w2 / n)

    delta_hidden = np.dot(delta_output, w2[:, :-1]) * hl_bias[:, :-1] * (1 - hl_bias[:, :-1])
    grad_w1 = np.dot(delta_hidden.T, training_data) / n + (lambdaval * w1 / n)

    # Flatten gradients
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()))

    return obj_val, obj_grad

def nnPredict(w1, w2, data):
    """
    % nnPredict predicts the label of data given the parameter w1, w2 of Neural Network.

    % Input:
    % w1: weights from input to hidden
    % w2: weights from hidden to output
    % data: input features

    % Output:
    % label: vector of predicted labels
    """

    labels = np.array([])

    # Adding bias to input data
    n_data = data.shape[0]
    input_bias = np.ones((n_data, 1))
    i_data = np.hstack((data, input_bias)) 

    # Hidden layer activation
    hidden_input = np.dot(i_data, w1.T)
    hidden_output = sigmoid(hidden_input)

    # Adding bias to hidden layer output
    hi_bias = np.ones((hidden_output.shape[0], 1))
    hl_bias = np.hstack((hidden_output, hi_bias))

    # Output layer activation
    ol_input = np.dot(hl_bias, w2.T)
    ol_output = sigmoid(ol_input)

    labels = np.argmax(ol_output, axis=1)

    return labels



"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')



    # Update Lambda and Hidden Units
    lambda_values = list(range(0,61,10))
    hidden_units_list = [4, 8, 12, 16, 20]

    results = {'lambda': [], 'hidden_units': [], 'train_accuracy': [],
              'validation_accuracy': [], 'test_accuracy': [], 'training_time': []}

    best_validation_accuracy = 0
    best_params = None
    selected_params = None

    

    for lambdaval, n_hidden in itertools.product(lambda_values, hidden_units_list):
        start_time = time.time()
        
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        opts = {'maxiter': 50}
        
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        training_time = time.time() - start_time
        
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
        
        train_accuracy = 100 * np.mean(nnPredict(w1, w2, train_data) == train_label)
        validation_accuracy = 100 * np.mean(nnPredict(w1, w2, validation_data) == validation_label)
        test_accuracy = 100 * np.mean(nnPredict(w1, w2, test_data) == test_label)
        
        results['lambda'].append(lambdaval)
        results['hidden_units'].append(n_hidden)
        results['train_accuracy'].append(train_accuracy)
        results['validation_accuracy'].append(validation_accuracy)
        results['test_accuracy'].append(test_accuracy)
        results['training_time'].append(training_time)

        # Printing accuracies for current lambda and hidden unit
        print(f"Lambda: {lambdaval}, Hidden Units: {n_hidden} -> "
          f"Train Acc: {train_accuracy:.2f}%, "
          f"Validation Acc: {validation_accuracy:.2f}%, "
          f"Test Acc: {test_accuracy:.2f}%, "
          f"Training Time: {training_time:.2f}s")
        
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_params = {
                'lambda': lambdaval, 'hidden_units': n_hidden,
                'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy,
                'test_accuracy': test_accuracy, 'training_time': training_time,
                'w1': w1, 'w2': w2}
            selected_params = {
                'hidden_units': n_hidden, 'w1': w1, 'w2': w2, 'lambda': lambdaval}

    # Save best parameters
    with open('params.pickle', 'wb') as f:
        pickle.dump(selected_params, f)


    # Print selected parameters
    print("\nBest Parameter:")
    print(f"Hidden Units: {selected_params['hidden_units']}")
    print(f"Lambda: {selected_params['lambda']}")
    print(f"Training Accuracy: {best_params['train_accuracy']:.2f}%")
    print(f"Validation Accuracy: {best_params['validation_accuracy']:.2f}%")
    print(f"Test Accuracy: {best_params['test_accuracy']:.2f}%")
    print(f"Training Time: {best_params['training_time']:.2f} seconds")


    # Graph Validation Accuracy vs Lambda
    plt.figure(figsize=(12, 8))
    for hidden_units in hidden_units_list:
        mask = np.array(results['hidden_units']) == hidden_units
        plt.plot(np.array(results['lambda'])[mask],
                np.array(results['validation_accuracy'])[mask],
                marker='o', label=f'Hidden Units = {hidden_units}')
    plt.xlabel('Lambda (λ) Value')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy vs Lambda for Different Hidden Units')
    plt.legend()
    plt.grid(True)
    plt.show()
