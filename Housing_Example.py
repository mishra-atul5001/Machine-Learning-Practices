import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

#load Data Set
from sklearn.datasets import load_boston
boston = load_boston()

x=boston.data
y=boston.target
# Fit the Model and Split the DATA
linear_regress = linear_model.LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
linear_regress.fit(x_train,y_train)

a = linear_regress.predict(x_test)

print(a)


# Checking the Accuracy Score..!!
# from sklearn.metrics import accuracy_score
# # print(accuracy_score(y_test,a))
# metrics.accuracy_score is used to measure classification accuracy, it can't be used 
# to measure accuracy of a regression model because it doesn't make sense to see
# accuracy for regression - predictions rarely can equal the expected values.

#Mean Squre Error
print('Mean Squared Error is: ')
mse=np.mean((a-y_test)**2)
print(mse)

# Hence we use metrics.r2_score to obtain regression score function

from sklearn.metrics import r2_score,accuracy_score
print(r2_score(y_test,a)*100) 
