import tensorflow as tf
from tensorflow import keras

#Different Types of Models in Keras....Using Tensorflow..!!

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(64,activation = 'relu'))
# Add another:
model.add(keras.layers.Dense(64,activation = 'relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10,activation='softmax'))
# Create a sigmoid layer:
model.add(keras.layers.Dense(64, activation='sigmoid'))
# Or:
model.add(keras.layers.Dense(64, activation=tf.sigmoid))
# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
model.add(keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01)))
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
model.add(keras.layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01)))
# A linear layer with a kernel initialized to a random orthogonal matrix:
model.add(keras.layers.Dense(64, kernel_initializer='orthogonal'))
# A linear layer with a bias vector initialized to 2.0s:
model.add(keras.layers.Dense(64, bias_initializer=keras.initializers.constant(2.0)))

#Count upto 5 in tensorflow
count = tf.Variable(0)
newVal = tf.math.add(count,1)
assignment = tf.assign(count,newVal)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

for count in range(5):
	print(sess.run(assignment))
