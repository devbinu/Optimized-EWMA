# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:11:33 2018

@author: HP
"""
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
import numpy as np

alpha  = np.random.random()
beta = np.random.random()
gamma = np.random.random()


def HW_Linear(series,s,n):
    m = len(series)
    L = float(0)
    for i in range(0,s):
        L = L + series[m-n+i]
    L = L/s
    b = float(0)
    for i in range(0,s):
        b = b + (series[m-n+s+i]-series[m-n+i])
    b = b/(s*s)
    S = np.zeros(m,float)
    for i in range(0,s):
        S[m-n+i] = series[m-n+i]-L
    f = float(0)
    for i in range(m-n+s,m):
        p = L
        L = alpha*(series[i]-S[i-s]) + (1-alpha)*(L+b)
        b = beta*(L-p)+(1-beta)*b
        S[i] = gamma*(series[i]-L) + (1-gamma)*S[i-s]
        f = L + b + S[i-s]
    return f

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import time
start_time = time.time()
def parser(x):
	return datetime.strptime(x, '%Y-%m')
 
series = read_csv('tree-italian-canyon-new-mexico-l.csv')
X = series.values
M = 1153
mIN = min(X[0:M])
mAX = max(X[0:M])
X = (X-mIN)/(mAX-mIN)
size = int(M* 0.7)
n = np.random.randint(10, size)
train, test = X[0:size], X[size:M]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    p = train
    yhat = HW_Linear(p,12,n)
    predictions.append(yhat)
    obs = test[t]
    train = np.append(train,test[t])
    history.append(obs)
    print('iteration=%d predicted=%f, expected=%f' % (t,yhat*(mAX-mIN)+mIN, obs*(mAX-mIN)+mIN))
error = mean_squared_error(test, predictions)
error1 = mean_absolute_error(test,predictions)
print('Window Size: %d Test MSE: %f' % (n,error))
print('Window Size: %d Test MAE: %f' % (n,error1))
print('Window Size: %d Test NMSE: %f' % (n,float(error/(max(test)[0]-min(test)[0]))))
print('Window Size: %d Test NMAE: %f' % (n,float(error1/(max(test)[0]-min(test)[0]))))
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
print("--- %s seconds ---" % (time.time() - start_time))


# Code for exporting data to Excel File
import xlrd, xlwt
w4 = xlwt.Workbook()
ws4 = w4.add_sheet('Sheet')

n = len(predictions)

ws4.write(0, 0, 'Predicted')
ws4.write(0, 1, 'Actual')

for i in range(n):
    p = float(predictions[i]*(mAX-mIN)+mIN)
    q = float(test[i]*(mAX-mIN)+mIN)
    ws4.write(i+1, 0, p)
    ws4.write(i+1, 1, q)

w4.save('.\Output\Tree\Holt_Winter_A.xls')

