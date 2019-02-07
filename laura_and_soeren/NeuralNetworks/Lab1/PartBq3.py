#
# Project 1, part B question 2
# Find the optimal learning rate for the 3 layer network using 5 fold learning
#

import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('Figures_B3'):
    print('creating the figures folder')
    os.makedirs('Figures_B3')

        ## Assgin to GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


NUM_FEATURES = 8
NUM_NEURONS = [20, 40, 60, 80, 100]
NUM_LABELS = 1

LEARNING_RATE = 10**(-9)
NUM_EXP = 5
FOLD = 5
BETA = 10**(-3)
EPOCHS = 200
BATCH_SIZE = 32
SEED = 255
np.random.seed(SEED)



def train_network(X, Y, Xt, Yt, neurons):
                # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])

        ## Build the graph for the deep net
                # Hidden layer
        w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, neurons],
                                             stddev=1.0 / np.sqrt(NUM_FEATURES),
                                             dtype=tf.float32),
                         name='weights')
        b1 = tf.Variable(tf.zeros([neurons]), dtype=tf.float32, name='biases')
        h = tf.nn.relu( tf.matmul(x, w1) + b1 )
                # Linear layer
        w2 = tf.Variable(tf.truncated_normal([neurons, NUM_LABELS],
                                             stddev=1.0 / np.sqrt(neurons),
                                             dtype=tf.float32),
                         name='weights')
        b2 = tf.Variable(tf.zeros([NUM_LABELS]), dtype=tf.float32, name='biases')
        y = tf.matmul(h, w2) + b2 

                # Loss with L2 regulation
        loss = tf.reduce_mean(tf.square(y_ - y)) + BETA*( tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) )

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
                        if i % 100 == 0:
                                print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))

                        if i == EPOCHS-1:
                                print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))

        return te_err
        #return np.vstack((tr_err, te_err))

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

        # experiment with small datasets
        #trainX = trainX[:1000]
        #trainY = trainY[:1000]

        trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
        testX = (testX - np.mean(trainX, axis=0))/ np.std(trainX, axis=0)


                ## Do the loops
        val_err_avg = []     # Average validation error
        val_err = []         # Validation error

        
        part = np.int( 1*trainX.shape[0]/FOLD ) # number of data to use for testing
        i = 0
        for neurons in NUM_NEURONS:
                        # Make a data set that  can be changed
                X_train, Y_train = trainX, trainY
                
                        # Do the fold
                err_fold = []
                val_err_fold = []   # Validation error
                
                
                for num_fold in range( FOLD ):
                        print('neurons: %d Fold: %d'%(neurons, num_fold))

                                # Take first part of data as test and last part as training
                        part_X_test, part_Y_test = X_train[:part], Y_train[:part]
                        part_X_train, part_Y_train = X_train[part:], Y_train[part:]

                        val_err_fold.append( train_network( part_X_train, part_Y_train, part_X_test, part_Y_test, neurons) )

                                # Place the test data in the end of the array
                        X_train = np.concatenate(( part_X_train, part_X_test),axis=0)
                        Y_train = np.concatenate(( part_Y_train, part_Y_test),axis=0)                


                val_err_avg.append( np.mean( val_err_fold, axis = 0 ) )
                val_err.append( val_err_fold  )



        ## Plot Cross-validation error

        for plotter in range( NUM_EXP ):
                                    
            fig = plt.figure(2*plotter)
            ax = fig.add_subplot(111)
            for fold in range( FOLD ):
                ax.plot( range(EPOCHS), val_err[plotter][fold], label = 'Fold %d'%fold)
            plt.xlabel('Epochs')
            plt.ylabel('Mean square error')
            plt.title('5 fold 32 batch with %d hidden neurons'%(NUM_NEURONS[plotter]))
            plt.legend()
            plt.savefig('Figures_B3/a%d_valdiationError.png'%(plotter))


            plt.figure(2*plotter + 1)
            plt.plot( range(EPOCHS), val_err_avg[plotter],  label = 'Average, neurons %g'%(NUM_NEURONS[plotter]))

            plt.xlabel('Epochs')
            plt.ylabel('Average Mean square error')
            plt.title('Average MSE Hidden Neurons:%g'%(NUM_NEURONS[plotter]))
            plt.legend()
            plt.savefig('Figures_B3/a%d_averagevaldiationError.png'%(plotter))



        ## Retrain the network with full data and test on test data
        i = 1
        for neurons in NUM_NEURONS:
            
            te_err = train_network( trainX, trainY, testX, testY, neurons)
            
                    # plot learning curves
            plt.figure(i + 2*NUM_EXP)
            plt.plot(range( EPOCHS ), te_err, label = 'test error')
            plt.xlabel('Epochs')
            plt.ylabel('Mean square error')
            plt.title('Test error with %d hidden neurons'%(neurons))
            plt.legend()
            plt.savefig('Figures_B3/b%d_testError.png'%i)
            i += 1

        #plt.show()



if __name__ == '__main__':
  main()
