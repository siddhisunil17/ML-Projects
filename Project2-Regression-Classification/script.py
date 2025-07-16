import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    classes = np.unique(y)
    k = len(classes)
    d = X.shape[-1]
    means = np.zeros([d, k])
    covmat = np.zeros([d, d])
    total_points = 0

    for i, cls in enumerate(classes):
        idx = (y.flatten() == cls)
        X_class = X[idx, :]
        n_class = X_class.shape[0]
        total_points = total_points + n_class
        means[:,i] = np.average(X_class, axis=0)
        center = X_class - means[:, i]
        covmat = covmat + center.T.dot(center)

    covmat /= (total_points - k)
    return means, covmat

def qdaLearn(X, y):
    classes = np.unique(y)
    k = len(classes)
    d = X.shape[-1]
    means = np.zeros([d, k])
    covmats = []

    for i, cls in enumerate(classes):
        X_class = X[y.flatten() == cls]
        means[:, i] = X_class.mean(0)
        center = X_class - means[:, i]
        class_covmat = center.T.dot(center)/(X_class.shape[0] - 1)
        covmats.append(class_covmat)

    return means, covmats

def ldaTest(means, covmat, Xtest, ytest):
    cov_inv = inv(covmat)
    k = means.shape[1]
    output_scores = np.zeros([Xtest.shape[0], k])
    for i in range(k):
        center = means[:, i]
        offset = Xtest - center
        output_scores[:, i] = -0.5 * np.sum((offset @ cov_inv) * offset, axis=1)

    ypred = (np.argmax(output_scores, axis=1) + 1).reshape(-1, 1)
    acc = np.mean(ypred == ytest)

    return acc, ypred

def qdaTest(means, covmats, Xtest, ytest):
    k = means.shape[1]
    classes = np.arange(1, k + 1)
    N = Xtest.shape[0]
    scores = np.zeros((N, k))

    for i in range(k):
        inverse_cov = inv(covmats[i])
        sign_val, logdet_val = np.linalg.slogdet(covmats[i])
        center = Xtest - means[:, i]
        scores[:, i] = -0.5*(logdet_val + np.sum((center@inverse_cov)*center,1))

    idx_pred = np.argmax(scores, axis=1)
    ypred = (classes[idx_pred]).reshape([-1,1])
    acc = np.mean(ypred == ytest)
    return acc, ypred

def learnOLERegression(X, y):
    XTX = np.dot(X.T, X)
    XTX_inv=np.linalg.inv(XTX)
    XTy = np.dot(X.T, y)
    w=np.dot(XTX_inv, XTy)
    return w

def testOLERegression(w, Xtest, ytest):
    y_pred = np.dot(Xtest, w)
    mse = np.sum((ytest - y_pred)**2) /Xtest.shape[0]
    return mse

def learnRidgeRegression(X, y, lambd):
    XtX = np.dot(X.T, X)
    penalty_matrix = np.eye(X.shape[1]) * lambd
    XTy = np.dot(X.T, y)
    w = np.linalg.inv(XtX + penalty_matrix).dot(XTy)
    return w

def regressionObjVal(w, X, y, lambd):
    if w.ndim > 1:
        w = w.flatten()
    w_col = w[:,None]

    ydiff = y - np.dot(X, w_col)
    error = 0.5 * (np.sum(ydiff**2) + lambd * np.sum(w_col**2))

    grad = - np.dot(X.T, ydiff) + lambd * w_col
    grad = grad.flatten()

    return error, grad

def mapNonLinear(x, p):
    if x.ndim == 1:
        x = x.reshape(-1,1)
    N = x.shape[0]
    Xp = np.ones([N, p+1])
    for i in range(1, p+1):
        Xp[:, i] = np.power(x[:,0], i)
    return Xp


# Main script
# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])

plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init.flatten(), jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.04 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()