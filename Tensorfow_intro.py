import tensorflow as tf
#from tensorflow import keras
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Checking Tensorflow
hello = tf.constant('Hello...Tensorflow...Yes it is Working..!!')
sess = tf.Session()
print(sess.run(hello))
#Use sess.run(hello).decode() because it is a bytestring.
#decode() method will return the string.
print(sess.run(hello).decode())
#sess.close()

#Addition of Numbers in Tensorflow to gain more Insight

num1 = tf.constant(5,name = "a")
num2 = tf.constant(25, name="b")
c_add = tf.math.add(num1,num2,name="c")
print('Value of c before: ',c_add)
sess1 = tf.Session()
output = sess1.run(c_add)
print('After Addition...Value is: ', output)
#sess1.close()

#Exercise : Generate a random array of size 100 and 
#find mean of z as obtained from below equations:
#1. y = x*x + 5x
#2. z = y*y

x=tf.random_normal([100],seed = 12,name = "x")
y = tf.Variable(x*x+5*x,name="y")
z= tf.multiply(y,y,name = "z")
z_mean = tf.reduce_mean(z)

init_op = tf.global_variables_initializer()

with tf.Session() as sess2:
	sess2.run(init_op)
	zout =sess2.run(z_mean)
	print("Value of Z is:",zout) 


hello = tf.constant('Hello...Tensorflow...Yes it is Working..!! Again..!!')
sess3 = tf.Session()
print(sess3.run(hello))
#Use sess.run(hello).decode() because it is a bytestring.
#decode() method will return the string.
print(sess3.run(hello).decode())

