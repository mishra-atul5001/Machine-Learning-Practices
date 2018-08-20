import sklearn
import numpy
import scipy
from sklearn import tree
print("Apple vs Orange")
print("1-> Orange and 0-> Apple")
features = [[140,1],[130,1],[150,0],[170,0]]
#140,150,130,170 are the weights
#1 for SMOOTH , 0 for BUMPY surface
labels = [0,0,1,1]
#0 for Apple and 1 for Orange
#We tarin the Model using Tree on features and labels
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
#Now let's see if it works for a Test Input
print(clf.predict([[160,0]]))




# Apple vs Orange
# 1-> Orange and 0-> Apple
# [1]
# [Finished in 0.7s]
# And it works perfectly fine..!!