#
# Project 1, part B question 4
# Find the optimal number of layers
#

import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('Figures_B4'):
    print('creating the figures folder')
    os.makedirs('Figures_B4')

        ## Assgin to GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


NUM_FEATURES = 8
NUM_L1_NEURONS = 20
NUM_L2_NEURONS = 20
NUM_L3_NEURONS = 20
NUM_LABELS = 1

LEARNING_RATE = 10**(-9)
KEEP_PROB = 0.9
NUM_EXP = 5
FOLD = 5
BETA = 10**(-3)
EPOCHS = 100
BATCH_SIZE = 32
SEED = 255
np.random.seed(SEED)



def train_4L_network(X, Y, Xt, Yt, dropout):
                # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])

        ## Build the graph for the deep net
                # L1 Hidden layer
        w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_L1_NEURONS],
                                             stddev=1.0 / np.sqrt(NUM_FEATURES),
                                             dtype=tf.float32),
                         name='weights')
        b1 = tf.Variable(tf.zeros([NUM_L1_NEURONS]), dtype=tf.float32, name='biases')
        if dropout == 1:
            h1 = tf.nn.relu( tf.matmul(x, w1) + b1 )
            h1_dropout = tf.nn.dropout(h1, KEEP_PROB)
        elif dropout == 0:
            h1 = tf.nn.relu( tf.matmul(x, w1) + b1 )
            

                        # L2 Hidden layer
        w2 = tf.Variable(tf.truncated_normal([NUM_L1_NEURONS, NUM_L2_NEURONS],
                                             stddev=1.0 / np.sqrt(NUM_L1_NEURONS),
                                             dtype=tf.float32),
                         name='weights')
        b2 = tf.Variable(tf.zeros([NUM_L2_NEURONS]), dtype=tf.float32, name='biases')
        if dropout == 1:
            h2 = tf.nn.relu( tf.matmul(h1_dropout, w2) + b2 )
            h2_dropout = tf.nn.dropout(h2, KEEP_PROB)
        elif dropout == 0:
            h2 = tf.nn.relu( tf.matmul(h1, w2) + b2 )
            
        
                # Linear layer
        w3 = tf.Variable(tf.truncated_normal([NUM_L2_NEURONS, NUM_LABELS],
                                             stddev=1.0 / np.sqrt(NUM_L2_NEURONS),
                                             dtype=tf.float32),
                         name='weights')
        b3 = tf.Variable(tf.zeros([NUM_LABELS]), dtype=tf.float32, name='biases')
        if dropout == 1:
            y = tf.matmul(h2_dropout, w3) + b3 
        elif dropout == 0:
            y = tf.matmul(h2, w3) + b3 
            
                # Loss with L2 regulation
        loss = tf.reduce_mean(tf.square(y_ - y))

        ## Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer( LEARNING_RATE )
        train_op = optimizer.minimize(loss)
        error = tf.reduce_mean(tf.square(y_ - y))


        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                idx = np.arange(X.shape[0])
                tr_err, te_err = [], []
                
                for i in range( EPOCHS ):
                                # Random mize data for mini batch                
                        np.random.shuffle(idx)
                        X, Y = X[idx], Y[idx]
                        
                        for p in range(X.shape[0]// BATCH_SIZE):
                                train_op.run(feed_dict=
                                             {x: X[p*BATCH_SIZE: (p+1)*BATCH_SIZE],
                                              y_: Y[p*BATCH_SIZE: (p+1)*BATCH_SIZE]})


                                # Test network after training
                        tr_err.append(error.eval(feed_dict={x: X, y_: Y}))
                                # Test on unseen data
                        te_err.append(error.eval(feed_dict={x: Xt, y_: Yt}))
                        if i % 10 == 0:
                                print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))

                        if i == EPOCHS-1:
                                print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))

        #return te_err
        return np.vstack((tr_err, te_err))
def train_5L_network(X, Y, Xt, Yt, dropout):
                # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])

        ## Build the graph for the deep net
                # L1 Hidden layer
        w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_L1_NEURONS],
                                             stddev=1.0 / np.sqrt(NUM_FEATURES),
                                             dtype=tf.float32),
                         name='weights')
        b1 = tf.Variable(tf.zeros([NUM_L1_NEURONS]), dtype=tf.float32, name='biases')
        if dropout == 1:
            h1 = tf.nn.relu( tf.matmul(x, w1) + b1 )
            h1_dropout = tf.nn.dropout(h1, KEEP_PROB)
        elif dropout == 0:
            h1 = tf.nn.relu( tf.matmul(x, w1) + b1 )
            

                        # L2 Hidden layer
        w2 = tf.Variable(tf.truncated_normal([NUM_L1_NEURONS, NUM_L2_NEURONS],
                                             stddev=1.0 / np.sqrt(NUM_L1_NEURONS),
                                             dtype=tf.float32),
                         name='weights')
        b2 = tf.Variable(tf.zeros([NUM_L2_NEURONS]), dtype=tf.float32, name='biases')
        if dropout == 1:
            h2 = tf.nn.relu( tf.matmul(h1_dropout, w2) + b2 )
            h2_dropout = tf.nn.dropout(h2, KEEP_PROB)
        elif dropout == 0:
            h2 = tf.nn.relu( tf.matmul(h1, w2) + b2 )

                        # L2 Hidden layer
        w3 = tf.Variable(tf.truncated_normal([NUM_L2_NEURONS, NUM_L3_NEURONS],
                                             stddev=1.0 / np.sqrt(NUM_L2_NEURONS),
                                             dtype=tf.float32),
                         name='weights')
        b3 = tf.Variable(tf.zeros([NUM_L3_NEURONS]), dtype=tf.float32, name='biases')
        if dropout == 1:
            h3 = tf.nn.relu( tf.matmul(h2_dropout, w3) + b3 )
            h3_dropout = tf.nn.dropout(h3, KEEP_PROB)
        elif dropout == 0:
            h3 = tf.nn.relu( tf.matmul(h2, w3) + b3 )
            
        
                # Linear layer
        w4 = tf.Variable(tf.truncated_normal([NUM_L3_NEURONS, NUM_LABELS],
                                             stddev=1.0 / np.sqrt(NUM_L3_NEURONS),
                                             dtype=tf.float32),
                         name='weights')
        b4 = tf.Variable(tf.zeros([NUM_LABELS]), dtype=tf.float32, name='biases')
        if dropout == 1:
            y = tf.matmul(h3_dropout, w4) + b4 
        elif dropout == 0:
            y = tf.matmul(h3, w4) + b4 
            
                # Loss with L2 regulation
        loss = tf.reduce_mean(tf.square(y_ - y))

        ## Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer( LEARNING_RATE )
        train_op = optimizer.minimize(loss)
        error = tf.reduce_mean(tf.square(y_ - y))


        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                idx = np.arange(X.shape[0])
                tr_err, te_err = [], []
                
                for i in range( EPOCHS ):
                                # Random mize data for mini batch                
                        np.random.shuffle(idx)
                        X, Y = X[idx], Y[idx]
                        
                        for p in range(X.shape[0]// BATCH_SIZE):
                                train_op.run(feed_dict=
                                             {x: X[p*BATCH_SIZE: (p+1)*BATCH_SIZE],
                                              y_: Y[p*BATCH_SIZE: (p+1)*BATCH_SIZE]})


                                # Test network after training
                        tr_err.append(error.eval(feed_dict={x: X, y_: Y}))
                                # Test on unseen data
                        te_err.append(error.eval(feed_dict={x: Xt, y_: Yt}))
                        if i % 10 == 0:
                                print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))

                        if i == EPOCHS-1:
                                print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))

        #return te_err
        return np.vstack((tr_err, te_err))

def main():
        ## Read and divide data into test and train sets 
        cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
        X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
        Y_data = (np.asmatrix(Y_data)).transpose()

        idx = np.arange(X_data.shape[0])
        np.random.shuffle(idx)
        X_data, Y_data = X_data[idx], Y_data[idx]

        m = 3* X_data.shape[0] // 10
        trainX, trainY = X_data[m:], Y_data[m:]
        testX, testY = X_data[:m], Y_data[:m]
        
        ## experiment with small datasets
        #trainX = trainX[:1000]
        #trainY = trainY[:1000]

        trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
        testX = (testX - np.mean(trainX, axis=0))/ np.std(trainX, axis=0)



        ## Do the loops
        val_err_4L = []     # Training error
        val_err_5L = []    # Validation error
        te_err_4L = []     # Training error
        te_err_5L = []    # Validation error

                # 4 Layer with (1) and without (0) dropout
        err_4L = train_4L_network( trainX, trainY, testX, testY, 0 ) 
        val_err_4L.append( err_4L[0] )
        te_err_4L.append( err_4L[1] )

        err_4L = train_4L_network( trainX, trainY, testX, testY, 1 )
        val_err_4L.append( err_4L[0] )
        te_err_4L.append( err_4L[1] )

                # 4 Layer with (1) and without (0) dropout
        err_5L = train_5L_network( trainX, trainY, testX, testY, 0 ) 
        val_err_5L.append( err_5L[0] )
        te_err_5L.append( err_5L[1] )
        
        err_5L = train_5L_network( trainX, trainY, testX, testY, 1 ) 
        val_err_5L.append( err_5L[0] )
        te_err_5L.append( err_5L[1] )



        ## Plot the data L4
        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)

        ax1.scatter( range( EPOCHS ), val_err_4L[0], marker="x", label='Without dropout')
        ax1.scatter( range( EPOCHS ), val_err_4L[1], marker="o", label='With dropout')
        #plt.legend(loc='upper left');
        plt.xlabel('Epochs')
        plt.ylabel('Training mean square error ')
        plt.title('Validation error vs Epochs 4 Layers')
        plt.legend()
        plt.savefig('Figures_B4/L4val.png')

        fig = plt.figure(2)
        ax1 = fig.add_subplot(111)

        ax1.scatter( range( EPOCHS ), te_err_4L[0], marker="x", label='Without dropout')
        ax1.scatter( range( EPOCHS ), te_err_4L[1], marker="o", label='With dropout')
        #plt.legend(loc='upper left');
        plt.xlabel('Epochs')
        plt.ylabel('Test error ')
        plt.title('Test error vs Epochs 4 Layers')
        plt.legend()
        plt.savefig('Figures_B4/L4test.png')




     
        ## Plot the data L5
        fig = plt.figure(3)
        ax1 = fig.add_subplot(111)

        ax1.scatter( range( EPOCHS ), val_err_5L[0], marker="x", label='Without dropout')
        ax1.scatter( range( EPOCHS ), val_err_5L[1], marker="o", label='With dropout')
        #plt.legend(loc='upper left');
        plt.xlabel('Epochs')
        plt.ylabel('Training mean square error ')
        plt.title('Validation error vs Epochs 5 Layers')
        plt.legend()
        plt.savefig('Figures_B4/L5a.png')

        fig = plt.figure(4)
        ax1 = fig.add_subplot(111)
        
        ax1.scatter( range( EPOCHS ), te_err_5L[0], marker="x", label='Without dropout')
        ax1.scatter( range( EPOCHS ), te_err_5L[1], marker="o", label='With dropout')
        #plt.legend(loc='upper left');
        plt.xlabel('Epochs')
        plt.ylabel('Test error ')
        plt.title('Test error vs Epochs 5 Layers')
        plt.legend()
        plt.savefig('Figures_B4/L5b.png')


        #plt.show()



if __name__ == '__main__':
  main()
