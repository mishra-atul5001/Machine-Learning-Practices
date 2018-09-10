import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt
#%matplotlib inline

no_samples = 100
x = np.linspace(-np.pi,np.pi,no_samples)
y=0.5*x + np.sin(x)+np.random.random(x.shape)
plt.scatter(x,y,color='black')
plt.xlabel('x - Input Feature')
plt.ylabel('y - Target Values')
plt.title('Data for Linear Regression..!!')
plt.show()

#Splitting the DATA
#Training set(75%)
#Validation set(15%)
#Test Set(15%)
random_indices = np.random.permutation(no_samples)
#Training Set 
x_train = x[random_indices[:70]] 
y_train=y[random_indices[:70]]
#Validation Set
x_val = x[random_indices[70:85]]
y_val = y[random_indices[70:85]]
#Test Set
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]

#fitting the LINE
model_linear = linear_model.LinearRegression()

x_train_fit = np.matrix(x_train.reshape(len(x_train),1))
y_train_fit = np.matrix(y_train.reshape(len(y_train),1))

#FITTING THE MODEL

model_linear.fit(x_train_fit,y_train_fit)

plt.scatter(x_train,y_train,color='blue')
plt.plot(x.reshape((len(x),1)),model_linear.predict(x.reshape(len(x),1)),color='green')
plt.xlabel('X - Input features')
plt.ylabel('Y - Output Features')
plt.title('Line Fitted with GREEN Color..!!')
plt.show()

#Evalute the MODEL With MSE 

mean_val_error = np.mean((y_val - model_linear.predict(x_val.reshape(len(x_val),1)))**2)
mean_test_error = np.mean((y_test - model_linear.predict(x_test.reshape(len(x_test),1)))**2)

print('Validation MSE:',  mean_val_error)
print('Test MSE:',  mean_test_error)

#Decision Tree Regression:
max_depth_tree= np.arange(10)+1
train_er = []
val_er = []
test_er = []
for depth in max_depth_tree:
	model = tree.DecisionTreeRegressor(max_depth = depth)
	x_train_for_tree = np.matrix(x_train.reshape(len(x_train),1))
	y_train_for_tree = np.matrix(y_train.reshape(len(y_train),1))

	#Fit the Line
	model.fit(x_train_for_tree,y_train_for_tree)

	#Plot the Line
	plt.figure()
	plt.scatter(x_train,y_train,color='blue')
	plt.plot(x.reshape((len(x),1)),model.predict(x.reshape((len(x),1))),color='green')
	plt.xlabel('x-input feature')
	plt.ylabel('y-target values')
	plt.title('Line fit to training data with max_depth='+str(depth))
	mean_train_error = np.mean( (y_train - model.predict(x_train.reshape(len(x_train),1)))**2 )
	mean_val_error = np.mean( (y_val - model.predict(x_val.reshape(len(x_val),1)))**2 )
	mean_test_error = np.mean( (y_test - model.predict(x_test.reshape(len(x_test),1)))**2 )
	train_er.append(mean_train_error)
	val_er.append(mean_val_error)
	test_er.append(mean_test_error)
	print('Training MSE: ', mean_train_error, '\nValidation MSE: ', mean_val_error, '\nTest MSE: ', mean_test_error)
    
plt.figure()
plt.plot(train_er,c='red')
plt.plot(val_er,c='blue')
plt.plot(test_er,c='green')
plt.legend(['Training error', 'Validation error', 'Test error'])
plt.title('Variation of error with maximum depth of tree')
plt.show()
plt.show()

#Decision Tree Classifier:
iris = datasets.load_iris()
X = iris.data #Choosing only the first two input-features
Y = iris.target

number_of_samples = len(Y)
#Splitting into training, validation and test sets
random_indices = np.random.permutation(number_of_samples)
#Training set
num_training_samples = int(number_of_samples*0.7)
x_train = X[random_indices[:num_training_samples]]
y_train = Y[random_indices[:num_training_samples]]
#Validation set
num_validation_samples = int(number_of_samples*0.15)
x_val = X[random_indices[num_training_samples : num_training_samples+num_validation_samples]]
y_val = Y[random_indices[num_training_samples: num_training_samples+num_validation_samples]]
#Test set
num_test_samples = int(number_of_samples*0.15)
x_test = X[random_indices[-num_test_samples:]]
y_test = Y[random_indices[-num_test_samples:]]

#Fit the Model

model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

#Visualize the DATA
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image

dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())

#Evalutae the MODEL
validation_set_predictions = [model.predict(x_val[i].reshape((1,len(x_val[i]))))[0] for i in range(x_val.shape[0])]
validation_misclassification_percentage = 0
for i in range(len(validation_set_predictions)):
    if validation_set_predictions[i]!=y_val[i]:
        validation_misclassification_percentage+=1
validation_misclassification_percentage *= 100/len(y_val)
print ('validation misclassification percentage =', validation_misclassification_percentage, '%')

test_set_predictions = [model.predict(x_test[i].reshape((1,len(x_test[i]))))[0] for i in range(x_test.shape[0])]

test_misclassification_percentage = 0
for i in range(len(test_set_predictions)):
    if test_set_predictions[i]!=y_test[i]:
        test_misclassification_percentage+=1
test_misclassification_percentage *= 100/len(y_test)
print ('test misclassification percentage =', test_misclassification_percentage, '%')