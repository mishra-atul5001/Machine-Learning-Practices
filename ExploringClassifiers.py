# importing datasets
from sklearn import datasets
iris = datasets.load_iris()
X= iris.data
y= iris.target
# Splitting Data using Cross-Validation
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= .5)
# Now Preparing the Classifier/Model
from sklearn import tree
my_classifier_tree = tree.DecisionTreeClassifier()
# Training out Tree Classifier
my_classifier_tree.fit(X_train,y_train)
# Getting predictions
predictions_tree = my_classifier_tree.predict(X_test)
# Checking the Accuracy of our Model
from sklearn.metrics import accuracy_score
print("Accuracy using DecisionTreeClassifier : ")
print(accuracy_score(y_test,predictions_tree))

# Now we will try to modify our Model using KNeighbour Classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier_KNeighbor = KNeighborsClassifier()
my_classifier_KNeighbor.fit(X_train,y_train)
# Getting predictions
predictions_Kneighbor = my_classifier_KNeighbor.predict(X_test)
# Checking the Accuracy of our Model
from sklearn.metrics import accuracy_score
print("Accuracy using KNeighbourClassifier : ")
print(accuracy_score(y_test,predictions_Kneighbor))

# Output:
# Accuracy using DecisionTreeClassifier : 
# 0.9466666666666667
# Accuracy using KNeighbourClassifier : 
# 0.9733333333333334
# Accuracy May change due to DataSet alteration, but the nearest accuracy remains SAME.

# The takeaway from here is that since there are many Different Classifiers, at higher levels, they have similar interface likes Lines:
# my_classifier_KNeighbor.fit(X_train,y_train)
# predictions_Kneighbor = my_classifier_KNeighbor.predict(X_test)


