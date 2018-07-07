import tflearn
import tensorflow as tf
import numpy as np
import tflearn.datasets.mnist as mnist
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow import reset_default_graph

tf.reset_default_graph()

##============================================================================


#784 inputs as image is 32x32, so 784
network = input_data(shape=[None, 784], name='input')

network = fully_connected(network, 64, activation="relu")
network = dropout(network, 0.8)

network = fully_connected(network, 64, activation="relu")
network = dropout(network, 0.8)

network = fully_connected(network, 10, activation="softmax")

network = regression(network, optimizer="adam", learning_rate=1e-3,
                    loss="categorical_crossentropy", name="targets")

model = tflearn.DNN(network, tensorboard_dir="log")



##============================================================================


#X: input
#Y: output

X, Y, testX, testY = mnist.load_data(one_hot=True)

if not model:
    model = nerual_network_model()

#tensorflow.reset_default_graph()
model.fit(X, Y, validation_set = (testX, testY), n_epoch=5, show_metric=True,
    run_id="number_regconition")


##==========================================================================

































##
