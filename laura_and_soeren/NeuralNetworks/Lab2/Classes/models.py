import numpy as np
import tensorflow as tf
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
MAX_LABEL = 15

BATCH_SIZE = 128
LR = 0.01

seed = 333
tf.set_random_seed(seed)
np.random.seed(seed)
tf.logging.set_verbosity(tf.logging.ERROR)

class early_stopper:

    def __init__(self, max_counter):
        '''
        max_counter: The amout of worse accuracy before the training stops
        '''

        self.best_acc = 0
        self.time_counter = 0
        self.max_time = max_counter

    def check_accuracy(self, acc_value):
        '''Use to see if early stopping should be applied.
           Returns 1 for early stopping
        '''
        if(self.best_acc < acc_value):
            self.timer_counter = 0
            self.best_acc = acc_value
        else:
            self.time_counter += 1

        if( self.time_counter > self.max_time):
            return 1

        return 0

           
class cnn_char_model:
    def __init__(self, N_Filters, Filter_Shape1, Filter_Shape2, Pool_Window, Pool_Stride, Dropout_Rate=[1,1,1]):
            ## Define the Network layers
        self.name      =  '_cnn_char_dr_' + str(Dropout_Rate[0]) + '_'+ str(Dropout_Rate[1]) + '_' + str(Dropout_Rate[2])
        self.N_filters = N_Filters
        self.filter_shape1 = Filter_Shape1
        self.filter_shape2 = Filter_Shape2
        self.pool_window   = Pool_Window
        self.pool_stride   = Pool_Stride
        self.dropout_rate  = Dropout_Rate

            ## Define the input
        self.x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        self.y_ = tf.placeholder(tf.int64)


            ## Optimizer - Do the the softmax cross enxtropy.
        self.logits = self.model()
        self.entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = tf.one_hot( self.y_, MAX_LABEL ), logits = self.logits
                )
            )
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.entropy)

        self.correct_predictions = tf.equal(
            tf.argmax( tf.nn.softmax( self.logits ),1),
            tf.argmax( tf.one_hot( self.y_, MAX_LABEL ),1) )
        
        self.correct_predictions = tf.cast(self.correct_predictions, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_predictions)


    def model(self):
        ## Construct the network 
        print('Create the char cnn model: ' + self.name)

        # Input is BATCH_SIZE x 100.   Onehot make it a BATCH_SIZE x 100 x 256 Matrix.
        # Reshape to BATCH_SIZE x 100 x 256 x 1
        #            BATCH_SIZE x HEIGHT x WIDTH x CHANNELS
        input_layer = tf.reshape(
            tf.one_hot(self.x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

        with tf.variable_scope('CNN_char_Layer1'):
            conv1 = tf.layers.conv2d(
                input_layer,
                filters     = self.N_filters,
                kernel_size = self.filter_shape1,
                padding     = 'VALID',
                activation  = tf.nn.relu
                )
                # Output BATCH_SIZE x 100-20+1 x 256-256+1
                #        BATCH_SIZE x 81 x 1 x 10

    
            if np.mean( self.dropout_rate ) != 1:
                conv1 = tf.layers.dropout(conv1, self.dropout_rate[0])            
    
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size   = self.pool_window,
                strides     = self.pool_stride,
                padding     = 'SAME'
                )
                # Output BATCH_SIZE x 41 x 1 x 10

        with tf.variable_scope('CNN_char_Layer2'):
            conv2 = tf.layers.conv2d(
                pool1,
                filters     = self.N_filters,
                kernel_size = self.filter_shape2,
                padding     = 'VALID',
                activation  = tf.nn.relu
                )
                # Output BATCH_SIZE x 41-20+1 x 1-1+1 x 10
                #        BATCH_SIZE x 22 x 1 x 10

            if np.mean( self.dropout_rate ) != 1:
                conv2 = tf.layers.dropout(conv2, self.dropout_rate[1])
    
            pool2 = tf.layers.max_pooling2d(
                conv2,
                pool_size   = self.pool_window,
                strides     = self.pool_stride,
                padding     = 'SAME'
                )
                # Output BATCH_SIZE x 11 x 1 x 10

                # Reduce_max take the maximumvalue form each Onehot vector (find the char), and squeze the [1] dimension
            pool2 = tf.squeeze( tf.reduce_max( pool2, 1 ), squeeze_dims = [1] )
                # Output BATCH_SIZE x 10

        # Create the onput, and do not use a sofmax function, because thie will be done later  
        logits = tf.layers.dense(
            pool2,
            MAX_LABEL,
            activation = None#tf.nn.softmax
            )
        if np.mean( self.dropout_rate ) != 1:
            logits = tf.layers.dropout(logits, self.dropout_rate[2])

        return logits   

    def train(self, x_train, y_train, x_test, y_test, epochs = 100, early_stop = 0):
            ## Train the network
        print('Train the char cnn model: ' + self.name)
        stp = early_stopper(early_stop)
        no_epochs = epochs
      
        with tf.Session() as sess:       
            sess.run(tf.global_variables_initializer())

            # training
            loss = []
            tr_err, te_acc = [], [] # Test error and training accuracy

            N = len(x_train)
            idx = np.arange(N)
            for e in range(no_epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
          

                    #Do the batch learning
                for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                    sess.run([self.train_op, self.entropy], {self.x: x_train[start:end], self.y_: y_train[start:end]})

                    # Find the error and accuracy
                tr_err.append( self.entropy.eval(feed_dict = {self.x: x_train, self.y_: y_train}) )  
                te_acc.append( self.accuracy.eval(feed_dict = {self.x: x_test, self.y_: y_test}) )

                if e%1 == 0:
                    print('iter: %d, entropy: %g, accuracy: %g'%(e, tr_err[e], te_acc[e]))

                
                    ## Implement early stopping
                if( stp.check_accuracy(te_acc[-1]) and early_stop != 0 ):
                    no_epochs = e+1     # update the number of epochs so the plotting works
                    break

            ## Save the plots        
        plt.figure()
        plt.plot(range(len(tr_err)), tr_err)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training error vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'train_Err.png')


        plt.figure()
        plt.plot(range(len(te_acc)), te_acc)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'test_acc.png')

        tf.reset_default_graph()

        return tr_err, te_acc


##################################################################################################
class cnn_word_model:
    def __init__(self, N_Filters, Filter_Shape1, Filter_Shape2, Pool_Window, Pool_Stride, N_Words, Dropout_Rate =[1, 1, 1]):

            ## Define the network layers
        self.name      =  '_cnn_word_dr_' + str(Dropout_Rate[0]) + '_'+ str(Dropout_Rate[1]) + '_' + str(Dropout_Rate[2])
        self.N_filters = N_Filters
        self.filter_shape1 = Filter_Shape1
        self.filter_shape2 = Filter_Shape2
        self.pool_window   = Pool_Window
        self.pool_stride   = Pool_Stride
        self.n_words       = N_Words
        self.dropout_rate  = Dropout_Rate

        self.x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        self.y_ = tf.placeholder(tf.int64)


                    # Optimizer - Do the the softmax cross enxtropy.
        self.logits = self.model()
        self.entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = tf.one_hot( self.y_, MAX_LABEL ), logits = self.logits
                )
            )
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.entropy)

        self.correct_predictions = tf.equal(
            tf.argmax( tf.nn.softmax( self.logits ),1),
            tf.argmax( tf.one_hot( self.y_, MAX_LABEL ),1) )
        
        self.correct_predictions = tf.cast(self.correct_predictions, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_predictions)

    def model(self):
            ## Build the network
        print('Chreate the word cnn model: ' + self.name)

            # Input is BATCH_SIZE x MAX_DOCUMENT_LENGTH.
            # Word vector is BATCH_SIZE x MAX_DOCUMENT_LENGTH x EMBEDDING SIZE
            # Reshape to BATCH_SIZE x 100 x 2 x 1
            #    BATCH_SIZE x HEIGHT x WIDTH x CHANNELS

        self.word_vectors = tf.reshape(
            tf.contrib.layers.embed_sequence( self.x, vocab_size=self.n_words, embed_dim = 20), [-1, MAX_DOCUMENT_LENGTH, 20, 1]
            )

        with tf.variable_scope('CNN_word_Layer1'):
            conv1 = tf.layers.conv2d(
                self.word_vectors,
                filters     = self.N_filters,
                kernel_size = self.filter_shape1,
                padding     = 'VALID',
                activation  = tf.nn.relu
                )
                # Output BATCH_SIZE x 100-20+1 x 20-20+1 x 10
                #        BATCH_SIZE x 81 x 1 x 10

    
            if np.mean( self.dropout_rate ) != 1:
                conv1 = tf.layers.dropout(conv1, self.dropout_rate[0]) 
                
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size   = self.pool_window,
                strides     = self.pool_stride,
                padding     = 'SAME'
                )
                # Output BATCH_SIZE x 41 x 1 x 10


        with tf.variable_scope('CNN_word_Layer2'):
            conv2 = tf.layers.conv2d(
                pool1,
                filters     = self.N_filters,
                kernel_size = self.filter_shape2,
                padding     = 'VALID',
                activation  = tf.nn.relu
                )
                # Output BATCH_SIZE x 41-20+1 x 1-1+1 x 10
                #        BATCH_SIZE x 22 x 1 x 10

    
            if np.mean( self.dropout_rate ) != 1:
                conv2 = tf.layers.dropout(conv2, self.dropout_rate[1]) 
    
            pool2 = tf.layers.max_pooling2d(
                conv2,
                pool_size   = self.pool_window,
                strides     = self.pool_stride,
                padding     = 'SAME'
                )
                # Output BATCH_SIZE x 11 x 1 x 10


                # Reduce_max take the maximumvalue form each Onehot vector (find the char), and squeze the [1] dimension
            pool2 = tf.squeeze( tf.reduce_max( pool2, 1 ), squeeze_dims = [1] )
                # Output BATCH_SIZE x 10

                # Create the onput, and do not use a sofmax function, because thie will be done later  
        logits = tf.layers.dense(
            pool2,
            MAX_LABEL,
            activation = None #tf.nn.softmax
            )

        if np.mean( self.dropout_rate ) != 1:
            logits = tf.layers.dropout(logits, self.dropout_rate[2])
            
        return logits



    def train(self, x_train, y_train, x_test, y_test, epochs = 100, early_stop = 0):
            ## Train the network
        print('Train the word cnn model: ' + self.name)

        stp = early_stopper(early_stop)
      
        with tf.Session() as sess:       
            sess.run(tf.global_variables_initializer())

            # training
            loss = []
            tr_err, te_acc = [], [] # Test error and training accuracy

            N = len(x_train)
            idx = np.arange(N)
            for e in range(epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
          

                    #Do the batch learning
                for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                    sess.run([self.train_op, self.entropy, self.word_vectors], {self.x: x_train[start:end], self.y_: y_train[start:end]})

                    # Find the error and accuracy
                tr_err.append( self.entropy.eval(feed_dict = {self.x: x_train, self.y_: y_train}) )  
                te_acc.append( self.accuracy.eval(feed_dict = {self.x: x_test, self.y_: y_test}) )

                if e%1 == 0:
                    print('iter: %d, entropy: %g, accuracy: %g'%(e, tr_err[e], te_acc[e]))


                    ## Implement early stopping
                if( stp.check_accuracy(te_acc[-1]) and early_stop != 0 ):
                    break

        ## Save the plots        
        plt.figure()
        plt.plot(range(len(tr_err)), tr_err)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training error vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'train_Err.png')


        plt.figure()
        plt.plot(range(len(te_acc)), te_acc)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'test_acc.png')

        tf.reset_default_graph()
        
        return tr_err, te_acc


######################################################################
class rnn_char_model:
    def __init__(self, Hidden_Size , Dropout_Rate= [1, 1, 1]):
        self.name        = 'RNN_char_dr_' + str(Dropout_Rate[0]) + '_'+ str(Dropout_Rate[1]) + '_' + str(Dropout_Rate[2])
        self.hidden_size = Hidden_Size
        self.dropout_rate= Dropout_Rate

            # Create the model
        self.x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        self.y_ = tf.placeholder(tf.int64)

        self.logits= self.model()
        
        self.entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y_, MAX_LABEL), logits=self.logits))
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.entropy)

        
        self.correct_predictions = tf.equal( tf.argmax(tf.nn.softmax(self.logits),1), tf.argmax(tf.one_hot( self.y_, MAX_LABEL ),1) )
        self.correct_predictions = tf.cast(self.correct_predictions, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_predictions)


    def model(self):
        print('Create the char rnn model: ' + self.name)

        input_layer = tf.reshape(
           tf.one_hot(self.x, 256), [-1, MAX_DOCUMENT_LENGTH, 256])

        word_list = tf.unstack(input_layer, axis=1)

        with tf.variable_scope('RNN_char_Layer'):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            if np.mean( self.dropout_rate ) != 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     input_keep_prob = self.dropout_rate[0],
                                                     output_keep_prob= self.dropout_rate[1],
                                                     state_keep_prob = self.dropout_rate[2]
                                                     )
                
            _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

            logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

        return logits



    def train(self, x_train, y_train, x_test, y_test, epochs = 100, early_stop = 0):  
        print('Train the char rnn model: ' + self.name)
        
        stp = early_stopper(early_stop)
      
        with tf.Session() as sess:       
            sess.run(tf.global_variables_initializer())

            # training
            loss = []
            tr_err, te_acc = [], [] # Test error and training accuracy

            N = len(x_train)
            idx = np.arange(N)
            for e in range(epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
          

                    #Do the batch learning
                for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                    sess.run([self.train_op, self.entropy], {self.x: x_train[start:end], self.y_: y_train[start:end]})
    
                    # Find the error and accuracy
                tr_err.append( self.entropy.eval(feed_dict = {self.x: x_train, self.y_: y_train}) )  
                te_acc.append( self.accuracy.eval(feed_dict = {self.x: x_test, self.y_: y_test}) )

                if e%1 == 0:
                    print('iter: %d, entropy: %g, accuracy: %g'%(e, tr_err[e], te_acc[e]))


                    ## Implement early stopping
                if( stp.check_accuracy(te_acc[-1])  and early_stop != 0 ):
                    break

                    ## Save the plots        
        plt.figure()
        plt.plot(range(len(tr_err)), tr_err)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training error vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'train_Err.png')


        plt.figure()
        plt.plot(range(len(te_acc)), te_acc)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'test_acc.png')

        tf.reset_default_graph()

        return tr_err, te_acc

###########################################################333
class rnn_word_model:
    def __init__(self, Hidden_Size, N_Words, Dropout_Rate= [1,1,1]):
        self.name        = 'RNN_word_dr_' + str(Dropout_Rate[0]) + '_'+ str(Dropout_Rate[1]) + '_' + str(Dropout_Rate[2])
        self.n_words     = N_Words
        self.hidden_size = Hidden_Size
        self.dropout_rate= Dropout_Rate

            # Create the model
        self.x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        self.y_ = tf.placeholder(tf.int64)

        self.logits, self.word_list = self.model()
        
        self.entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y_, MAX_LABEL), logits=self.logits))
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.entropy)

        
        self.correct_predictions = tf.equal( tf.argmax(tf.nn.softmax(self.logits),1), tf.argmax(tf.one_hot( self.y_, MAX_LABEL ),1) )
        self.correct_predictions = tf.cast(self.correct_predictions, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_predictions)

        

    def model(self):
        print('Create the word rnn model: ' + self.name)
       
        with tf.variable_scope('RNN_word_Layer'):

            word_vectors = tf.contrib.layers.embed_sequence(
                self.x, vocab_size=self.n_words, embed_dim=20)

            word_list = tf.unstack(word_vectors, axis=1)

            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            if np.mean( self.dropout_rate ) != 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     input_keep_prob = self.dropout_rate[0],
                                                     output_keep_prob= self.dropout_rate[1],
                                                     state_keep_prob = self.dropout_rate[2]
                                                     )
                
            _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

            logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

        return logits, word_list

    def train(self, x_train, y_train, x_test, y_test, epochs = 200, early_stop = 0):  
        print('Train the word rnn model: ' + self.name)
        stp = early_stopper(early_stop)
      
        with tf.Session() as sess:       
            sess.run(tf.global_variables_initializer())

            # training
            loss = []
            tr_err, te_acc = [], [] # Test error and training accuracy

            N = len(x_train)
            idx = np.arange(N)
            for e in range(epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
          

                    #Do the batch learning
                for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                    sess.run([self.train_op, self.entropy, self.word_list], {self.x: x_train[start:end], self.y_: y_train[start:end]})
    
                    # Find the error and accuracy
                tr_err.append( self.entropy.eval(feed_dict = {self.x: x_train, self.y_: y_train}) )  
                te_acc.append( self.accuracy.eval(feed_dict = {self.x: x_test, self.y_: y_test}) )

                if e%1 == 0:
                    print('iter: %d, entropy: %g, accuracy: %g'%(e, tr_err[e], te_acc[e]))


                    ## Implement early stopping
                if( stp.check_accuracy(te_acc[-1])  and early_stop != 0 ):
                    break

                
            ## Save the plots        
        plt.figure()
        plt.plot(range(len(tr_err)), tr_err)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training error vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'train_Err.png')


        plt.figure()
        plt.plot(range(len(te_acc)), te_acc)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'test_acc.png')

        tf.reset_default_graph()
        
        return tr_err, te_acc

################## VANILLA / LSTM #########################################333
class rnnx_word_model:
    def __init__(self, Hidden_Size, N_Words):
        self.name        = 'RNNx_word_dr_'
        self.n_words     = N_Words
        self.hidden_size = Hidden_Size

            # Create the model
        self.x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        self.y_ = tf.placeholder(tf.int64)

        self.logits, self.word_list = self.model()
        
        self.entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y_, MAX_LABEL), logits=self.logits))
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.entropy)

        
        self.correct_predictions = tf.equal( tf.argmax(tf.nn.softmax(self.logits),1), tf.argmax(tf.one_hot( self.y_, MAX_LABEL ),1) )
        self.correct_predictions = tf.cast(self.correct_predictions, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_predictions)

        

    def model(self):
        print('Create the word rnn model: ' + self.name)
       
        with tf.variable_scope('RNN_word_Layer'):

            word_vectors = tf.contrib.layers.embed_sequence(
                self.x, vocab_size=self.n_words, embed_dim=20)
            word_list = tf.unstack(word_vectors, axis=1)
        
            cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)
            cells_ = []
            for layer in range(2):
                cells_.append(cell)
            cells = tf.nn.rnn_cell.MultiRNNCell(cells_)
                
            _, encoding = tf.nn.static_rnn(cells, word_list, dtype=tf.float32)

            
            
            logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

        return logits, word_list

    def train(self, x_train, y_train, x_test, y_test, epochs = 200, early_stop = 0):  
        print('Train the word rnn model: ' + self.name)
        stp = early_stopper(early_stop)
      
        with tf.Session() as sess:       
            sess.run(tf.global_variables_initializer())

            # training
            loss = []
            tr_err, te_acc = [], [] # Test error and training accuracy

            N = len(x_train)
            idx = np.arange(N)
            for e in range(epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
          

                    #Do the batch learning
                for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                    sess.run([self.train_op, self.entropy, self.word_list], {self.x: x_train[start:end], self.y_: y_train[start:end]})
    
                    # Find the error and accuracy
                tr_err.append( self.entropy.eval(feed_dict = {self.x: x_train, self.y_: y_train}) )  
                te_acc.append( self.accuracy.eval(feed_dict = {self.x: x_test, self.y_: y_test}) )

                if e%1 == 0:
                    print('iter: %d, entropy: %g, accuracy: %g'%(e, tr_err[e], te_acc[e]))


                    ## Implement early stopping
                if( stp.check_accuracy(te_acc[-1])  and early_stop != 0 ):
                    break

                
            ## Save the plots        
        plt.figure()
        plt.plot(range(len(tr_err)), tr_err)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training error vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'train_Err.png')


        plt.figure()
        plt.plot(range(len(te_acc)), te_acc)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.savefig('Figures_B5/' + self.name + 'test_acc.png')

        tf.reset_default_graph()
        
        return tr_err, te_acc
