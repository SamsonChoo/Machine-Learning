import pandas as pd
import csv
import scipy
import numpy as np
import struct as st
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import random
from sklearn.linear_model import LogisticRegression     

def univariateGenerator(expectation, variance):
    u = scipy.random.uniform()
    v = scipy.random.uniform()
    
    z = scipy.sqrt(-2.0*scipy.log(u))*scipy.cos(2.0*scipy.pi*v)
    x = scipy.sqrt(variance)*z + expectation
#    w = scipy.sqrt(-2.0*scipy.log(u))*scipy.sin(2.0*scipy.pi*v)
#    y = scipy.sqrt(variance)*w + expectation
    return x

def genData(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
    with open('hw4regression.csv', mode = 'w') as data:
        writer = csv.writer(data,delimiter = ',',quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['label','x','y'])
        a,b = n,n
        while a!=0 and b !=0:
            k = random.randint(0,1)
            if k == 0 and a >0:
                writer.writerow([0,univariateGenerator(mx1,vx1), univariateGenerator(my1,vy1)])
                a -=1
            elif k ==1 and b > 0:
                writer.writerow([1,univariateGenerator(mx2,vx2), univariateGenerator(my2,vy2)])            
                b-=1
def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b 
def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result
def model_optimize(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost
def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred
def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w,b,X,Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #
        
        if (i % 100 == 0):
            costs.append(cost)
            #print("Cost after %i iteration is %f" %(i, cost))
    
    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return coeff, gradient, costs
def LR():
    col_names = ['label','x','y']
    data = pd.read_csv("hw4regression.csv",header=0, names=col_names) 
    feature_cols =['x','y']
    X = data[feature_cols]
    y = data.label
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    logreg = LogisticRegression()
    X_tr_arr = X_train
    X_ts_arr = X_test
    y_tr_arr = y_train.as_matrix()
    y_ts_arr = y_test.as_matrix()
    #Get number of features
    n_features = X_tr_arr.shape[1]
    print('Number of Features', n_features)
    w, b = weightInitialization(n_features)
    #Gradient Descent
    coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=10000)
    #Final prediction
    w = coeff["w"]
    b = coeff["b"]
    print('Optimized weights', w)
    print('Optimized intercept',b)
    #
#    final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
    final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)
    #
#    m_tr =  X_tr_arr.shape[0]
    m_ts =  X_ts_arr.shape[0]
#    y_tr_pred = predict(final_train_pred,m_tr)
    y_ts_pred = predict(final_test_pred, m_ts)

    cnf_matrix = metrics.confusion_matrix(y_ts_pred.T, y_ts_arr)
    print(cnf_matrix)
    print("Test Accuracy:",metrics.accuracy_score(y_ts_pred.T, y_ts_arr))
    print("Precision:",metrics.precision_score(y_ts_pred.T, y_ts_arr))
    print("Sensitivity:",cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[0][1]))
    print("Specificity:",cnf_matrix[1][1]/(cnf_matrix[1][0]+cnf_matrix[1][1]))
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    #
    y_pred=logreg.predict(X_test)
    print("==Ski-learn==")
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    #sensitivity = The proportion of observed positives that were predicted to be positive 
    print("Sensitivity:",cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[0][1]))
    print("Specificity:",cnf_matrix[1][1]/(cnf_matrix[1][0]+cnf_matrix[1][1]))
#    specificity = The proportion of observed negatives that were predicted to be negatives.
LR()
#genData(1000,1,1,1,1,0,0.5,0,0.5)
