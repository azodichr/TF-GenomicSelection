"""

Run the following code in HPC to:
    load linuxbrew
    load modules required by tensorflow
    activate the tensorflow python environment
    
source /opt/software/tensorflow/1.1.0/load_tf

python TF_MLP_2layer.py -x geno.csv -y pheno.csv -label Yld_Env1 -cv CVFs.csv -JobID 2
"""




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd

print("Test of TensorFlow install with Python 3.5.  This tests regression on GS data")
# FUNCTIONS

def multilayer_perceptron(x, weights, biases, activation_function):
    if activation_function == 'relu':
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    elif activation_function == 'sigmoid':
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.sigmoid(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.sigmoid(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer



hello = tf.constant('Hello, TensorFlow!')


#### Set default values #####
LABEL = 'Class'
activation_function = 'sigmoid'
training_epochs = 1000
n_hidden_1 = 100
n_hidden_2 = 50
learning_rate = 0.01
beta = 0.01  # regularization parameter 

for i in range (1,len(sys.argv),2):
  if sys.argv[i] == "-df":
    DF = sys.argv[i+1]
  if sys.argv[i] == "-x":
    X_file = sys.argv[i+1]
  if sys.argv[i] == "-y":
    Y_file = sys.argv[i+1]
  if sys.argv[i] == "-cv":
    CVs = sys.argv[i+1]
  if sys.argv[i] == "-JobID":
    JobID = int(sys.argv[i+1])    
  if sys.argv[i] == "-label":
    LABEL = sys.argv[i+1]
  if sys.argv[i] == "-act":
    activation_function = sys.argv[i+1] 
  if sys.argv[i] == "-epochs":
    training_epochs = int(sys.argv[i+1])
  if sys.argv[i] == "-l1":
    n_hidden_1 = int(sys.argv[i+1])
  if sys.argv[i] == "-l2":
    n_hidden_2 = int(sys.argv[i+1])
  if sys.argv[i] == "-lr":
    learning_rate = float(sys.argv[i+1])
  if sys.argv[i] == "-beta":
    beta = float(sys.argv[i+1])

# Read in geno and pheno and remove non target phenotypes
x = pd.read_csv(X_file, sep=',', index_col = 0)
y = pd.read_csv(Y_file, sep=',', index_col = 0)
y = y[[LABEL]]
yhat = np.zeros(shape = y.shape)

cv_folds = pd.read_csv(CVs, sep=',', index_col=0)
cv = cv_folds['cv_' + str(JobID-1)]
num_cvs = np.ptp(cv) + 1  # Range of values in cv (PeakToPeak)
training_error = []

for i in range(1,num_cvs+1):
    print("Predicting cv fold %i" % i)
    X_train = x[cv != i]
    X_test = x[cv == i]
    y_train = y[cv != i]
    y_test = y[cv == i]

    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    n_input = X_train.shape[1]
    n_samples = X_train.shape[0]
    n_classes = y_train.shape[1]


    # TF Graph Input
    nn_x = tf.placeholder(tf.float32, [None, n_input])
    nn_y = tf.placeholder(tf.float32, [None, n_classes])


    # Store layers weight & bias (default: mean=0, sd = 1)
    weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

    biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(nn_x, weights, biases, activation_function)

    # Define loss and optimizer
    # Cross entropy (for classifiers)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=nn_y))
    # Mean squared error
    loss = tf.reduce_mean(tf.square(pred - nn_y))
    regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])
    # Re-define loss with the regularization added
    loss = tf.reduce_mean(loss + beta * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    # Launch the graph
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict = {nn_x:X_train, nn_y:y_train})
        c = sess.run(loss,feed_dict = {nn_x:X_train, nn_y:y_train})

        if (epoch+1) % 250 == 0:
            print("Epoch:", '%04d' % (epoch+1), "Cost=", "{:.9f}".format(c))

        if epoch+1 == training_epochs:
            training_error.append(c)
            print('Final mse for training cv_%i: %.5f' % (i, c))
    # Predict test set and add to yhat output
    y_pred = sess.run(pred, feed_dict={nn_x: X_test})
    yhat[cv == i] = y_pred


print('Training error (MSE +/- stdev): %0.5f (%0.5f)' % (np.mean(training_error), np.std(training_error)))
testing_mse = np.mean((np.array(y)[:,0] - yhat[:,0])**2)
print('Testing error (MSE): %0.5f' % testing_mse)

cor = np.corrcoef(np.array(y)[:,0], yhat[:,0])
print('Accuracy (i.e. correlation coef): %.5f' % cor[0,1])


final = pd.merge(y, pd.DataFrame(yhat, index=x.index),right_index=True, left_index=True)
final.columns = ['y','yhat']
final.to_csv('cv_'+ str(JobID-1) + '.csv', sep=',', index=False)



