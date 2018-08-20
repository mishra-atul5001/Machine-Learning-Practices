# Iris Data Set Example

# Goals:
# 1. Import Data Set
# 2. Train a Classifier
# 3. Predict Label for a New Flavour
# 4. Visualize the TREE.
from sklearn.externals.six import StringIO
import pydot
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
iris= load_iris()

print (iris.feature_names)
print (iris.target_names)

# Output
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# ['setosa' 'versicolor' 'virginica']

print(iris.data[0])
print(iris.target[0])

# Output
# [5.1 3.5 1.4 0.2]
# 0

# printing the entire dataset:

for x in range(len(iris.target)):
	print("Example %d: Label %s: , features %s "%(x,iris.target[x],iris.data[x]))


# Example 0: Label 0: , features [5.1 3.5 1.4 0.2] 
# Example 1: Label 0: , features [4.9 3.  1.4 0.2] 
# Example 2: Label 0: , features [4.7 3.2 1.3 0.2] 
# Example 3: Label 0: , features [4.6 3.1 1.5 0.2] 
# Example 4: Label 0: , features [5.  3.6 1.4 0.2] 
# Example 5: Label 0: , features [5.4 3.9 1.7 0.4] 
# Example 6: Label 0: , features [4.6 3.4 1.4 0.3] 
# Example 7: Label 0: , features [5.  3.4 1.5 0.2] 
# Example 8: Label 0: , features [4.4 2.9 1.4 0.2] 
# Example 9: Label 0: , features [4.9 3.1 1.5 0.1] 
# Example 10: Label 0: , features [5.4 3.7 1.5 0.2] 
# Example 11: Label 0: , features [4.8 3.4 1.6 0.2] 
# # Example 12: Label 0: , features [4.8 3.  1.4 0.1] 
# Example 13: Label 0: , features [4.3 3.  1.1 0.1] 
# Example 14: Label 0: , features [5.8 4.  1.2 0.2] 
# Example 15: Label 0: , features [5.7 4.4 1.5 0.4] 
# Example 16: Label 0: , features [5.4 3.9 1.3 0.4] 
# Example 17: Label 0: , features [5.1 3.5 1.4 0.3] 
# Example 18: Label 0: , features [5.7 3.8 1.7 0.3] 
# Example 19: Label 0: , features [5.1 3.8 1.5 0.3] 
# Example 20: Label 0: , features [5.4 3.4 1.7 0.2] 
# Example 21: Label 0: , features [5.1 3.7 1.5 0.4] 
# Example 22: Label 0: , features [4.6 3.6 1.  0.2] 
# Example 23: Label 0: , features [5.1 3.3 1.7 0.5] 
# Example 24: Label 0: , features [4.8 3.4 1.9 0.2] 
# Example 25: Label 0: , features [5.  3.  1.6 0.2] 
# Example 26: Label 0: , features [5.  3.4 1.6 0.4] 
# Example 27: Label 0: , features [5.2 3.5 1.5 0.2] 
# Example 28: Label 0: , features [5.2 3.4 1.4 0.2] 
# Example 29: Label 0: , features [4.7 3.2 1.6 0.2] 
# Example 30: Label 0: , features [4.8 3.1 1.6 0.2] 
# Example 31: Label 0: , features [5.4 3.4 1.5 0.4] 
# Example 32: Label 0: , features [5.2 4.1 1.5 0.1] 
# Example 33: Label 0: , features [5.5 4.2 1.4 0.2] 
# Example 34: Label 0: , features [4.9 3.1 1.5 0.1] 
# Example 35: Label 0: , features [5.  3.2 1.2 0.2] 
# Example 36: Label 0: , features [5.5 3.5 1.3 0.2] 
# Example 37: Label 0: , features [4.9 3.1 1.5 0.1] 
# Example 38: Label 0: , features [4.4 3.  1.3 0.2] 
# Example 39: Label 0: , features [5.1 3.4 1.5 0.2] 
# Example 40: Label 0: , features [5.  3.5 1.3 0.3] 
# Example 41: Label 0: , features [4.5 2.3 1.3 0.3] 
# Example 42: Label 0: , features [4.4 3.2 1.3 0.2] 
# Example 43: Label 0: , features [5.  3.5 1.6 0.6] 
# Example 44: Label 0: , features [5.1 3.8 1.9 0.4] 
# Example 45: Label 0: , features [4.8 3.  1.4 0.3] 
# Example 46: Label 0: , features [5.1 3.8 1.6 0.2] 
# Example 47: Label 0: , features [4.6 3.2 1.4 0.2] 
# Example 48: Label 0: , features [5.3 3.7 1.5 0.2] 
# Example 49: Label 0: , features [5.  3.3 1.4 0.2] 
# Example 50: Label 1: , features [7.  3.2 4.7 1.4] 
# Example 51: Label 1: , features [6.4 3.2 4.5 1.5] 
# Example 52: Label 1: , features [6.9 3.1 4.9 1.5] 
# Example 53: Label 1: , features [5.5 2.3 4.  1.3] 
# Example 54: Label 1: , features [6.5 2.8 4.6 1.5] 
# Example 55: Label 1: , features [5.7 2.8 4.5 1.3] 
# Example 56: Label 1: , features [6.3 3.3 4.7 1.6] 
# Example 57: Label 1: , features [4.9 2.4 3.3 1. ] 
# Example 58: Label 1: , features [6.6 2.9 4.6 1.3] 
# Example 59: Label 1: , features [5.2 2.7 3.9 1.4] 
# Example 60: Label 1: , features [5.  2.  3.5 1. ] 
# Example 61: Label 1: , features [5.9 3.  4.2 1.5] 
# Example 62: Label 1: , features [6.  2.2 4.  1. ] 
# Example 63: Label 1: , features [6.1 2.9 4.7 1.4] 
# Example 64: Label 1: , features [5.6 2.9 3.6 1.3] 
# Example 65: Label 1: , features [6.7 3.1 4.4 1.4] 
# Example 66: Label 1: , features [5.6 3.  4.5 1.5] 
# Example 67: Label 1: , features [5.8 2.7 4.1 1. ] 
# Example 68: Label 1: , features [6.2 2.2 4.5 1.5] 
# Example 69: Label 1: , features [5.6 2.5 3.9 1.1] 
# Example 70: Label 1: , features [5.9 3.2 4.8 1.8] 
# Example 71: Label 1: , features [6.1 2.8 4.  1.3] 
# Example 72: Label 1: , features [6.3 2.5 4.9 1.5] 
# Example 73: Label 1: , features [6.1 2.8 4.7 1.2] 
# Example 74: Label 1: , features [6.4 2.9 4.3 1.3] 
# Example 75: Label 1: , features [6.6 3.  4.4 1.4] 
# # Example 76: Label 1: , features [6.8 2.8 4.8 1.4] 
# Example 77: Label 1: , features [6.7 3.  5.  1.7] 
# Example 78: Label 1: , features [6.  2.9 4.5 1.5] 
# Example 79: Label 1: , features [5.7 2.6 3.5 1. ] 
# Example 80: Label 1: , features [5.5 2.4 3.8 1.1] 
# Example 81: Label 1: , features [5.5 2.4 3.7 1. ] 
# Example 82: Label 1: , features [5.8 2.7 3.9 1.2] 
# Example 83: Label 1: , features [6.  2.7 5.1 1.6] 
# Example 84: Label 1: , features [5.4 3.  4.5 1.5] 
# Example 85: Label 1: , features [6.  3.4 4.5 1.6] 
# Example 86: Label 1: , features [6.7 3.1 4.7 1.5] 
# Example 87: Label 1: , features [6.3 2.3 4.4 1.3] 
# Example 88: Label 1: , features [5.6 3.  4.1 1.3] 
# Example 89: Label 1: , features [5.5 2.5 4.  1.3] 
# Example 90: Label 1: , features [5.5 2.6 4.4 1.2] 
# Example 91: Label 1: , features [6.1 3.  4.6 1.4] 
# Example 92: Label 1: , features [5.8 2.6 4.  1.2] 
# Example 93: Label 1: , features [5.  2.3 3.3 1. ] 
# Example 94: Label 1: , features [5.6 2.7 4.2 1.3] 
# Example 95: Label 1: , features [5.7 3.  4.2 1.2] 
# Example 96: Label 1: , features [5.7 2.9 4.2 1.3] 
# Example 97: Label 1: , features [6.2 2.9 4.3 1.3] 
# Example 98: Label 1: , features [5.1 2.5 3.  1.1] 
# Example 99: Label 1: , features [5.7 2.8 4.1 1.3] 
# Example 100: Label 2: , features [6.3 3.3 6.  2.5] 
# Example 101: Label 2: , features [5.8 2.7 5.1 1.9] 
# Example 102: Label 2: , features [7.1 3.  5.9 2.1] 
# Example 103: Label 2: , features [6.3 2.9 5.6 1.8] 
# Example 104: Label 2: , features [6.5 3.  5.8 2.2] 
# Example 105: Label 2: , features [7.6 3.  6.6 2.1] 
# Example 106: Label 2: , features [4.9 2.5 4.5 1.7] 
# Example 107: Label 2: , features [7.3 2.9 6.3 1.8] 
# Example 108: Label 2: , features [6.7 2.5 5.8 1.8] 
# Example 109: Label 2: , features [7.2 3.6 6.1 2.5] 
# Example 110: Label 2: , features [6.5 3.2 5.1 2. ] 
# Example 111: Label 2: , features [6.4 2.7 5.3 1.9] 
# Example 112: Label 2: , features [6.8 3.  5.5 2.1] 
# Example 113: Label 2: , features [5.7 2.5 5.  2. ] 
# Example 114: Label 2: , features [5.8 2.8 5.1 2.4] 
# Example 115: Label 2: , features [6.4 3.2 5.3 2.3] 
# Example 116: Label 2: , features [6.5 3.  5.5 1.8] 
# Example 117: Label 2: , features [7.7 3.8 6.7 2.2] 
# Example 118: Label 2: , features [7.7 2.6 6.9 2.3] 
# Example 119: Label 2: , features [6.  2.2 5.  1.5] 
# Example 120: Label 2: , features [6.9 3.2 5.7 2.3] 
# Example 121: Label 2: , features [5.6 2.8 4.9 2. ] 
# Example 122: Label 2: , features [7.7 2.8 6.7 2. ] 
# Example 123: Label 2: , features [6.3 2.7 4.9 1.8] 
# Example 124: Label 2: , features [6.7 3.3 5.7 2.1] 
# Example 125: Label 2: , features [7.2 3.2 6.  1.8] 
# Example 126: Label 2: , features [6.2 2.8 4.8 1.8] 
# Example 127: Label 2: , features [6.1 3.  4.9 1.8] 
# Example 128: Label 2: , features [6.4 2.8 5.6 2.1] 
# Example 129: Label 2: , features [7.2 3.  5.8 1.6] 
# Example 130: Label 2: , features [7.4 2.8 6.1 1.9] 
# Example 131: Label 2: , features [7.9 3.8 6.4 2. ] 
# Example 132: Label 2: , features [6.4 2.8 5.6 2.2] 
# Example 133: Label 2: , features [6.3 2.8 5.1 1.5] 
# Example 134: Label 2: , features [6.1 2.6 5.6 1.4] 
# Example 135: Label 2: , features [7.7 3.  6.1 2.3] 
# Example 136: Label 2: , features [6.3 3.4 5.6 2.4] 
# Example 137: Label 2: , features [6.4 3.1 5.5 1.8] 
# Example 138: Label 2: , features [6.  3.  4.8 1.8] 
# Example 139: Label 2: , features [6.9 3.1 5.4 2.1] 
# Example 140: Label 2: , features [6.7 3.1 5.6 2.4] 
# Example 141: Label 2: , features [6.9 3.1 5.1 2.3] 
# Example 142: Label 2: , features [5.8 2.7 5.1 1.9] 
# Example 143: Label 2: , features [6.8 3.2 5.9 2.3] 
# Example 144: Label 2: , features [6.7 3.3 5.7 2.5] 
# Example 145: Label 2: , features [6.7 3.  5.2 2.3] 
# Example 146: Label 2: , features [6.3 2.5 5.  1.9] 
# Example 147: Label 2: , features [6.5 3.  5.2 2. ] 
# Example 148: Label 2: , features [6.2 3.4 5.4 2.3] 
# Example 149: Label 2: , features [5.9 3.  5.1 1.8] 	


# Training Data and Testing Data:
test_idx = [0,50,100]
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

#Creating the Classifier:

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print(test_target)
print(clf.predict(test_data))

# Output:
# [0 1 2]
# [0 1 2]
# Since it is Same, hence we can see that the trained classifier works properly.

# VIZ CODE:
dot_data = StringIO()
tree.export_graphviz(clf,
	out_file = dot_data,
	feature_names = iris.feature_names,
	class_names = iris.target_names,
	filled = True,
	rounded=True,
	impurity = False
	)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")