
import numpy as np
import tflearn
from tflearn.datasets import mnist
MNIST_data = mnist.read_data_sets(one_hot=True)

data_train = MNIST_data.train
data_validation = MNIST_data.validation
data_test = MNIST_data.test
X, y = data_train._images, data_train._labels

tflearn.init_graph(num_cores=4)

net = tflearn.input_data(shape=[None, 784])
net = tflearn.fully_connected(net, 100, activation='relu')
net = tflearn.fully_connected(net, 100, activation='relu')
net = tflearn.fully_connected(net, 10, activation='softmax')

net = tflearn.regression(net, loss='categorical_crossentropy', optimizer='adam')

model = tflearn.DNN(net)

model.fit(X, y, n_epoch=1, batch_size=10, show_metric=True)


