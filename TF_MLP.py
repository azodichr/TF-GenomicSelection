"""
PURPOSE:
Fully connected MLP Neural Network Regression implemented in TensorFlow (TF)

INPUTS:
    REQUIRED:
    -x      File with genotype information
    -y      File with values you want to predict 
    -label  Name of column in y with the value you want to predict
    -save   Name to include in RESULTS file (i.e. what dataset are you running)
    -cv     File with CV folds specified
    -JobID  Which cv fold from the cv file do you want to run?

    OPTIONAL:
    -arc    Desired NN architecture as comma separated layer sizes (i.e. 100,50 or 200,200,50)
    -act    What activation function to use (sigmoid (default), relu, elu)
    -epochs Number of epochs to train on (default = 1000)
    -lr     Learning rate (default = 0.01)
    -beta   Regularization parameter (default = 0.01)


EXAMPLE ON HPCC:
Log on to development node with GPUs:
$ ssh dev-intel16-k80   
Load linuxbrew, modules required by TF, & activate the TF python environment
$ source /opt/software/tensorflow/1.1.0/load_tf
Run example MLP:
$ python TF_MLP.py -x geno.csv -y pheno.csv -label Yld_Env1 -cv CVFs.csv -JobID 2 -arc 100,50,20

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
import timeit

start_time = timeit.default_timer()
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# FUNCTIONS
def multilayer_perceptron(x, weights, biases, layer_number, activation_function):
    layer = x
    for l in range(1,layer_number+1):
        weight_name = 'h' + str(l)
        bias_name = 'b' + str(l)
        layer = tf.add(tf.matmul(layer, weights[weight_name]), biases[bias_name])
        if activation_function.lower() == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        elif activation_function.lower() == 'relu':
            layer = tf.nn.relu(layer)
        elif activation_function.lower() == 'elu':
            layer = tf.nn.elu(layer)
        else:
            print("Given activation function is not supported")
            quit()   
    out_layer = tf.matmul(layer, weights['out']) + biases['out']

    return out_layer


#### Set default values #####
activation_function = 'sigmoid'
training_epochs = 1000
arc = 100,50
learning_rate = 0.01
beta = 0.01  # regularization parameter 
SAVE = 'test'

for i in range (1,len(sys.argv),2):
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
  if sys.argv[i] == "-save":
    SAVE = sys.argv[i+1]
  if sys.argv[i] == "-act":
    activation_function = sys.argv[i+1] 
  if sys.argv[i] == "-epochs":
    training_epochs = int(sys.argv[i+1])
  if sys.argv[i] == "-lr":
    learning_rate = float(sys.argv[i+1])
  if sys.argv[i] == "-beta":
    beta = float(sys.argv[i+1])
  if sys.argv[i] == "-arc":     # Desired layer sizes comma separated (i.e. 100,50,20)
    arc = sys.argv[i+1]

# Read in the desired architecture
arc = arc.strip().split(',')
archit = []
for a in arc:
  archit.append(int(a))
layer_number = len(archit)

# Read in geno and pheno and remove non target phenotypes
x = pd.read_csv(X_file, sep=',', index_col = 0)
y = pd.read_csv(Y_file, sep=',', index_col = 0)
y = y[[LABEL]]
yhat = np.zeros(shape = y.shape)

cv_folds = pd.read_csv(CVs, sep=',', index_col=0)
cv = cv_folds['cv_' + str(JobID)]
num_cvs = np.ptp(cv) + 1  # Range of values in cv (PeakToPeak)
training_error = []

for i in range(1,num_cvs+1):
    print("Predicting cv fold %i" % i)
    X_train = x[cv != i]
    X_test = x[cv == i]
    y_train = y[cv != i]
    y_test = y[cv == i]

    n_input = X_train.shape[1]
    n_samples = X_train.shape[0]
    n_classes = y_train.shape[1]


    # TF Graph Input
    nn_x = tf.placeholder(tf.float32, [None, n_input])
    nn_y = tf.placeholder(tf.float32, [None, n_classes])


    # Store layers weight & bias (default: mean=0, sd = 1)
    weights = {}
    biases = {}
    weights['h1'] = tf.Variable(tf.random_normal([n_input, archit[0]]))
    biases['b1'] = tf.Variable(tf.random_normal([archit[0]]))
    for l in range(1,layer_number):
        w_name = 'h' + str(l+1)
        b_name = 'b' + str(l+1)
        weights[w_name] = tf.Variable(tf.random_normal([archit[l-1], archit[l]]))
        biases[b_name] = tf.Variable(tf.random_normal([archit[l]]))
    weights['out'] = tf.Variable(tf.random_normal([archit[-1], n_classes]))
    biases['out'] = tf.Variable(tf.random_normal([n_classes]))
    

    # Construct model
    pred = multilayer_perceptron(nn_x, weights, biases, layer_number, activation_function)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.square(pred - nn_y))   # Mean squared error
    regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])
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


testing_mse = np.mean((np.array(y)[:,0] - yhat[:,0])**2)
cor = np.corrcoef(np.array(y)[:,0], yhat[:,0])

stop_time = timeit.default_timer()

print('###################\nRESULTS\n###################\n')
print('Training error (MSE +/- stdev): %0.5f (%0.5f)' % (np.mean(training_error), np.std(training_error)))
print('Testing error (MSE): %0.5f' % testing_mse)
print('Accuracy (i.e. correlation coef): %.5f' % cor[0,1])
print('\nRun time: %s' % str(stop_time - start_time))


#final = pd.merge(y, pd.DataFrame(yhat, index=x.index),right_index=True, left_index=True)
#final.columns = ['y','yhat']
#final.to_csv('cv_'+ str(JobID-1) + '.csv', sep=',', index=False)

if not os.path.isfile('RESULTS.txt'):
    out2 = open('RESULTS.txt', 'a')
    out2.write('DateTime\tDF\tCVfold\tNumHidLay\tArchit\tActFun\tEpochs\tLearnRate\tBeta\tTrainError\tTrainErrorSTD\tTestError\tAccuracy\n')

out2 = open('RESULTS.txt', 'a')
out2.write('%s\t%s\t%s\t%i\t%s\t%s\t%i\t%0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\n' % (
    timestamp, SAVE, JobID, layer_number, arc, activation_function, training_epochs, learning_rate, beta,
     np.mean(training_error), np.std(training_error), testing_mse, cor[0,1]))
