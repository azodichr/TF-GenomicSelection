Tensor Flow: MLP for genomic selection

Goal: Use genetic marker information to predict a plant’s phenotype.

Overview of Steps

Data Pre-Processing
1.    Load genotype (x) and phenotype (y) data into pandas data frame
2.    Initialize an empty yhat array to populate with predicted training values later
3.    Determine how you will divide your data for training and testing
a.    I’m using predefined cross validation folds (CVFs.csv)
b.    You could also generate CV folds using KFold or StratifiedKFold
c.    With enough data you could take a train, validation, test approach instead.

Build a Model using each subset of training data & apply it to the testing data
1.    Define the placeholders for x & y
2.    Store the weights and biases for each layer
3.    Construct the MLP model, specify an activation function (relu is considered the standard act.fun., but sigmoid works much better in my case)
4.    Define the loss and optimizer
a.    Loss: For a regression NN, use MSE instead of softmax_cross_entropy
b.    Add a regularization function to the loss to prevent over-fitting the model. I used the l2 penalty (default beta =0.01)
c.    Optimizer: AdamOptimizer (learning rate default = 0.01)
5.    Launch the graph and prepare to initialize the variables
6.    For n epochs
a.    Evaluate the model on all of the training data (sess.run())
b.    Report the final MSE (MSE for last epoch) = training error. 
7.    Use the final model to predict the values from the testing set 
8.    Set yhat[test] equal to the predicted values

Output Model Metrics
1.    Calculate the training error: average final MSE over all cv folds
2.    Calculate the testing error: MSE between y & yhat
3.    Calculate accuracy (correlation coef) between y & yhat

Example

$ git clone git@github.com:azodichr/TF-GenomicSelection.git
$ source /opt/software/tensorflow/1.1.0/load_tf
$ cd TF-GenomicSelection/
$ python TF_MLP_2layer.py -x geno.csv -y pheno.csv -label Yld_Env1 -cv CVFs.csv -JobID 2 -act sigmoid -hl1 100 -hl2 50 -epochs 1000 -beta 0.01 -lr 0.01

