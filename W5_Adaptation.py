# Note: This file is from the week 5 workshop with minor additions

import tensorflow as tf
import numpy as np

from Constants import *

################################################################################

class DatasetIterator:
    """
    An iterator that returns randomized batches
    from a data set (with features and labels)
    """
    def __init__(self, features, labels):
        assert(features.shape[0] == labels.shape[0])
        assert(BATCH_SIZE > 0 and BATCH_SIZE <= features.shape[0])
        self.features = features
        self.labels = labels
        self.num_instances = features.shape[0]
        self.num_batches = self.num_instances // BATCH_SIZE
        if (self.num_instances % BATCH_SIZE != 0):
            self.num_batches += 1
        self._i = 0
        self._rand_ids = None

    def __iter__(self):
        self._i = 0
        self._rand_ids = np.random.permutation(self.num_instances)
        return self
        
    def __next__(self):
        if self.num_instances - self._i >= BATCH_SIZE:
            this_rand_ids = self._rand_ids[self._i:self._i + BATCH_SIZE]
            self._i += BATCH_SIZE
            return self.features[this_rand_ids], self.labels[this_rand_ids]
        elif self.num_instances - self._i > 0:
            this_rand_ids = self._rand_ids[self._i::]
            self._i = self.num_instances
            return self.features[this_rand_ids], self.labels[this_rand_ids]
        else:
            raise StopIteration()

################################################################################

# Runs a neural network in TensorFlow to make predictions on the data
def runNN(xTrain, yTrain, xTest, hidden_layers = []):

    # The number of features and labels in the given data
    num_features = len(xTrain[0])
    num_classes = len(list(set(yTrain)))

    # A binary classifier can be represented with one output neuron
    if (num_classes == 2):
        num_classes = 1

    xTrain = np.asarray(xTrain)
    yTrain = np.asarray(yTrain)
    xTest = np.asarray(xTest)
    
    train_iterator = DatasetIterator(xTrain, yTrain)

    X = tf.placeholder(dtype=tf.float32,
                       shape=[None, num_features], name="features")
    Y = tf.placeholder(dtype=tf.uint8,
                       shape=[None,], name="labels")

    with tf.variable_scope("neural-network", reuse=tf.AUTO_REUSE):

        inputs = num_features
        outputs = num_classes
        layers = [inputs] + hidden_layers + [outputs]
        
        # Create a matrix of weights for each layer
        W = [tf.get_variable("weights-" + str(i),
                             shape=[layers[i], layers[i + 1]],
                             dtype=tf.float32,
                             initializer=tf.glorot_uniform_initializer())
                             for i in range(len(layers) - 1)]

        # Create a list of biases for each layer (one bias per neuron)
        b = [tf.get_variable("bias-" + str(i),
                             shape=[layers[i + 1]],
                             dtype=tf.float32,
                             initializer=tf.glorot_uniform_initializer())
                             for i in range(len(layers) - 1)]

        # Feed-forward
        activations = [None for i in range(len(layers))]
        activations[0] = X
        for i in range(0, len(activations) - 1):
            weighted_sum = tf.add(tf.matmul(activations[i], W[i]), b[i])
            activations[i + 1] = tf.nn.sigmoid(weighted_sum)

        '''
        Note: No softmax is applied, so this neural
        network can only perform binary predictions
        '''
        
        # Flatten from [[a], [b], [c]] to [a, b, c]
        Y_pred = tf.reshape(activations[-1], [-1])
        
        loss = tf.losses.mean_squared_error(Y, Y_pred)
    
    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    opt_operation = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start = timer()
        
        # Run gradient descent for multiple epochs
        for epoch in range(EPOCHS):
            avg_loss = 0
            for X_batch, Y_batch in train_iterator:
                _, l = sess.run([opt_operation, loss], \
                                  feed_dict = {X: X_batch, Y: Y_batch})
                avg_loss += l / train_iterator.num_batches
            current = timer()
            print("Epoch {} / {}: loss = {:.4f} ({:.2f} secs)"
                  .format(epoch + 1, EPOCHS, avg_loss, current - start))
        print("Optimization complete.")

        # Make predictions
        predictions = Y_pred.eval({X: xTest, Y: []})
        
    return list(predictions)

################################################################################
