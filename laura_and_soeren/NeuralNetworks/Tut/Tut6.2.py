import tensorflow as tf
import numpy as np
import pylab as plt

## Define parameters
no_features = 2
no_labels = 1
no_neurons = [2, 4, 6, 8, 10]
#no_exp = 10
no_data = 11*11
fold = 5
part = np.int( 1*no_data/fold ) # number of data to use for testing

lr = 0.05
iters = 50
SEED = 10

np.random.seed(SEED)
tf.set_random_seed(SEED)


## Calculate the ffn(placeholder x, neurons)
def ffn(x, no_neurons):

        ## Define hidden layer dataflow
    with tf.name_scope('hidden'):
        weight = tf.Variable(
            tf.truncated_normal( [no_features, no_neurons],
                                 stddev= 1.0 / np.sqrt( float( no_features ))),
            name='weight')

        biases = tf.Variable( tf.zeros( [no_neurons] ), name='biases' )

        z = tf.matmul( x, weight ) + biases
        h = tf.sigmoid(z)
    
        ## Define output layer dataflow
    with tf.name_scope('output'):
        weight = tf.Variable(
            tf.truncated_normal([no_neurons, no_labels],
                                stddev = 1.0 / np.sqrt( float( no_neurons ))),
            name = 'weight' )
        
        biases = tf.Variable( tf.zeros( [no_labels] ), name='biases' )

        u = tf.matmul( h, weight ) + biases
        y = u 

    return y

## Calculate the optimize function
def train_exp(X, Y):
        # Split the data in train and test, X fold way    

    x_train, y_train = X[ part: ], Y[ part: ]
    x_test, y_test = X[ :part ], Y[ :part ]

    err = []
        
    ## Calculate error and optimize for each number of neurons
    for neurons_hidden in no_neurons:
        
        x = tf.placeholder( tf.float32, [None, no_features] )
        d = tf.placeholder( tf.float32, [None, no_labels] )

        y = ffn(x, neurons_hidden)

        loss = tf.reduce_mean( tf.reduce_sum( tf.square(d - y), axis = 1))
        train = tf.train.GradientDescentOptimizer( lr ).minimize( loss )
    # optimizer

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range (iters):
                train.run( feed_dict= {x:x_train, d:y_train} )
            err.append( loss.eval( feed_dict={x:x_test, d:y_test} ) )
    return err



## Main loop - build training data
def main():

        # Generate training data
    X_data = np.zeros((no_data, no_features))
    Y_data = np.zeros((no_data, no_labels))

    i=0
    for x1 in np.arange(-1., 1.02, 0.2):
        for x2 in np.arange(-1., 1.02, 0.2):
            X_data[i] = [x1,x2]
            Y_data[i] = np.sin(np.pi*x1)*np.cos(2*np.pi*x2)
            i += 1

        # Shuffle the data once
    idx = np.arange(no_data)
    np.random.shuffle(idx)
    X_data, Y_data = X_data[idx], Y_data[idx]

            ##Run the experimence
    err = []
    for exp in range(fold):
        print('exp %d'%exp)
        
        err.append( train_exp(X_data, Y_data) )

            ## Take first part of data as test and last part as training
        X_test, Y_test = X_data[:part], Y_data[:part]
        X_train, Y_train = X_data[part:], Y_data[part:]

            ## Place the test data in the end of the array
        X_data = np.concatenate(( X_train, X_test),axis=0)
        Y_data = np.concatenate((Y_train,Y_test),axis=0)



        # Print the mean errors of different models
    mean_err = np.mean(np.array(err), axis=0)
    print(mean_err)
    
    plt.figure(1)
    plt.plot( no_neurons, mean_err, marker = 'x', linestyle = 'None')
    plt.xticks( no_neurons)
    plt.xlabel('number of hidden neurons')
    plt.ylabel('mean error')

    print(' *hidden units* %d '%no_neurons[np.argmin( mean_err )])

    plt.show()

if __name__ == '__main__':
    main()
