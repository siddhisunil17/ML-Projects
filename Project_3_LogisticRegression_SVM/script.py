import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    x = np.hstack((np.ones((n_data, 1)), train_data))

    w = initialWeights.reshape(-1, 1)

    probs = sigmoid(x.dot(w))

    error = -np.mean(labeli * np.log(probs) + (1 - labeli) * np.log(1 - probs))

    error_grad = x.T.dot(probs - labeli) / n_data

    error_grad = error_grad.flatten()

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    n_data = data.shape[0]
    x = np.hstack((np.ones((n_data, 1)), data))

    class_probs = sigmoid(x.dot(W))

    label = np.argmax(class_probs, axis=1).reshape(-1, 1)

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    x = np.hstack((np.ones((n_data, 1)), train_data))

    W = params.reshape((n_feature + 1, labeli.shape[1]))

    class_scores = x.dot(W)
    class_scores -= np.max(class_scores , axis=1, keepdims=True)
    exp_scores =  np.exp(class_scores )
    class_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    error = -np.sum(labeli * np.log(class_probs)) / n_data
    error_grad = x.T.dot(class_probs - labeli) / n_data

    error_grad = error_grad.flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]

    x = np.hstack((np.ones((n_data, 1)), data))

    class_scores = x.dot(W)
    label = np.argmax(class_scores, axis=1).reshape(-1, 1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))


# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
svm_train_labels      = train_label.ravel()
svm_validation_labels = validation_label.ravel()
svm_test_labels       = test_label.ravel()

print("SVM with Linear Kernel")
linear_svm_model = SVC(kernel='linear')
linear_svm_model.fit(train_data, svm_train_labels)
print('Training Accuracy:', linear_svm_model.score(train_data, svm_train_labels) * 100)
print('Validation Accuracy:', linear_svm_model.score(validation_data, svm_validation_labels) * 100)
print('Test Accuracy:', linear_svm_model.score(test_data, svm_test_labels) * 100)

print("\nSVM with Radial Basis Function and gamma=1")
rbf1_svm_model = SVC(kernel='rbf', gamma=1.0)
rbf1_svm_model.fit(train_data, svm_train_labels)
print('Training Accuracy:', rbf1_svm_model.score(train_data, svm_train_labels) * 100)
print('Validation Accuracy:', rbf1_svm_model.score(validation_data, svm_validation_labels) * 100)
print('Test Accuracy:', rbf1_svm_model.score(test_data, svm_test_labels) * 100)

print("\nSVM with Radial Basis Function and default gamma")
rbf_default_svm = SVC(kernel='rbf')
rbf_default_svm.fit(train_data, svm_train_labels)
print('Training Accuracy:', rbf_default_svm.score(train_data, svm_train_labels) * 100)
print('Validation Accuracy:', rbf_default_svm.score(validation_data, svm_validation_labels) * 100)
print('Test Accuracy:', rbf_default_svm.score(test_data, svm_test_labels) * 100)

validation_accuracies = []
test_accuracies       = []

print("\nSVM with Radial Basis Function and varying C:")
C_values = [1,10,20,30,40,50,60,70,80,90,100]
for C in C_values:
    svm_c_model = SVC(C=C, kernel='rbf')
    svm_c_model.fit(train_data, svm_train_labels)
    val_acc  = svm_c_model.score(validation_data, svm_validation_labels) * 100
    test_acc = svm_c_model.score(test_data, svm_test_labels) * 100
    validation_accuracies.append(val_acc)
    test_accuracies.append(test_acc)
    print(f'C={C:3d}:- Train Acc: {svm_c_model.score(train_data, svm_train_labels)*100:.2f}%, '
          f'Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%')

# Plot accuracy vs C
plt.figure(figsize=(10, 6))
plt.plot(C_values, validation_accuracies, marker='o', label='Validation Accuracy')
plt.plot(C_values, test_accuracies,       marker='s', label='Test Accuracy')
plt.xlabel("C value")
plt.ylabel("Accuracy (%)")
plt.title("SVM RBF Kernel: Accuracy vs C")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1)*10)
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))


# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
