import numpy as np
import pylab as plt
import pandas
import tensorflow as tf
import csv
import timeit

from Classes.models import cnn_char_model
from Classes.models import rnn_char_model

from Classes.models import cnn_word_model
from Classes.models import rnn_word_model

from Classes.models import rnnx_word_model

import os
if not os.path.isdir('Figures_B5'):
    print('creating the figures folder')
    os.makedirs('Figures_B5')


MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
Q1_FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

Q2_FILTER_SHAPE1 = [20, 20]


Q3_HIDDEN_SIZE = 20


#MAX_LABEL = 15

#EARLY_STOPPER = 10

#tf.logging.set_verbosity(tf.logging.ERROR)
#seed = 10
#tf.set_random_seed(seed)
#np.random.seed(seed)



def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

def read_data_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words


  
def main():
  x_train_c, y_train_c, x_test_c, y_test_c = read_data_chars()
  x_train_w, y_train_w, x_test_w, y_test_w, no_words = read_data_words()


  #LSTM = SeriesPredictor()
  


##  word_rnn_model = rnn_word_model( Hidden_Size = Q3_HIDDEN_SIZE,
##                                   N_Words = no_words,
##                                   )
##
##  word_rnn_model.train( x_train_w,
##                        y_train_w,
##                        x_test_w,
##                        y_test_w,
##                        early_stop = 15
##                        )

  #LSTM = rnnx_word_model(Hidden_Size = 20, N_Words= no_words, Type = 'LSTM')




##  time_ = []
##
########################### CNN CHAR #################################
##
##  char_cnn_model = cnn_char_model(N_Filters = N_FILTERS,
##                                     Filter_Shape1 = Q1_FILTER_SHAPE1,
##                                     Filter_Shape2 = FILTER_SHAPE2,
##                                     Pool_Window = POOLING_WINDOW,
##                                     Pool_Stride = POOLING_STRIDE
##                                     )
##  start = timeit.default_timer()
##  tr_err_1, te_acc_1 = char_cnn_model.train( x_train_c,
##                                             y_train_c,
##                                             x_test_c,
##                                             y_test_c,
##                                             early_stop = 5
##                                             )
##  
##  stop = timeit.default_timer()
##  time_.append(stop-start)
##  print('Time: ', stop - start)
##
##  
########################### CNN WORD ############################
##  word_cnn_model = cnn_word_model(N_Filters = N_FILTERS,
##                                     Filter_Shape1 = Q2_FILTER_SHAPE1,
##                                     Filter_Shape2 = FILTER_SHAPE2,
##                                     Pool_Window = POOLING_WINDOW,
##                                     Pool_Stride = POOLING_STRIDE,
##                                     N_Words = no_words,
##                                     Dropout_Rate = [1,0.9,0.9]
##                                     )
##
##  word_cnn_model.train( x_train_w,
##                         y_train_w,
##                         x_test_w,
##                         y_test_w,
##                         early_stop = 15
##                         )



####### 

                         
##
################################# RNN CHAR ##############################
##
##  char_rnn_model = rnn_char_model( Hidden_Size = Q3_HIDDEN_SIZE)
##  start = timeit.default_timer()  
##  tr_err_3, te_acc_3 = char_rnn_model.train( x_train_c,
##                                                          y_train_c,
##                                                          x_test_c,
##                                                          y_test_c,
##                                                          early_stop = 5
##                                                          )
##  stop = timeit.default_timer()
##  time_.append(stop-start)
##  print('Time: ', stop - start)
##
################################ RNN WORD ###############################
##
##
##  word_rnn_model = rnn_word_model( Hidden_Size = Q3_HIDDEN_SIZE,
##                                   N_Words = no_words,
##                                   )
##  start = timeit.default_timer()
##  word_rnn_model.train( x_train_w,
##                        y_train_w,
##                        x_test_w,
##                        y_test_w,
##                        early_stop = 15
##                        )
##
##
##  stop = timeit.default_timer()
##  time_.append(stop-start)
##  print('Time: ', stop - start)

  

##  word_rnn_model1 = rnn_word_model( Hidden_Size = Q3_HIDDEN_SIZE,
##                                   N_Words = no_words,
##                                   Dropout_Rate = [1, 0.8, 1]
##                                   )
##  word_rnn_model1.train( x_train_w,
##                        y_train_w,
##                        x_test_w,
##                        y_test_w #,                        early_stop = 15
##                        )
##
##  word_rnn_model2 = rnn_word_model( Hidden_Size = Q3_HIDDEN_SIZE,
##                                   N_Words = no_words,
##                                   Dropout_Rate = [0.9, 0.9, 1]
##                                   )
##  word_rnn_model2.train( x_train_w,
##                        y_train_w,
##                        x_test_w,
##                        y_test_w #,                        early_stop = 15
##                        )
##
##  word_rnn_model3 = rnn_word_model( Hidden_Size = Q3_HIDDEN_SIZE,
##                                   N_Words = no_words,
##                                   Dropout_Rate = [0.9, 0.8, 1]
##                                   )
##  word_rnn_model3.train( x_train_w,
##                        y_train_w,
##                        x_test_w,
##                        y_test_w #,                        early_stop = 15
##                        )
##
##
##  word_rnn_model4 = rnn_word_model( Hidden_Size = Q3_HIDDEN_SIZE,
##                                   N_Words = no_words,
##                                   Dropout_Rate = [0.8, 0.8, 1]
##                                   )
##  word_rnn_model4.train( x_train_w,
##                        y_train_w,
##                        x_test_w,
##                        y_test_w#,                        early_stop = 15
##                        )

  

  ## Print

##  for i in range(len(time_)):
##    print(time_[i])
##
##  plt.figure()
##  plt.scatter(range(len(time_)), time_)
##  plt.xlabel('0:char CNN, 1:word CNN, 2:char RNN, 3:word RNN')
##  plt.ylabel('Time')
##  plt.title('Training time for the 4 networks')
##  plt.savefig('Figures_B5/timing.png')

##  plt.figure(1)
##  plt.plot(range(len(tr_err_1)), tr_err_1)
##  plt.xlabel('Iterations')
##  plt.ylabel('Error')
##  plt.title('Training error vs Iterations')
##  plt.savefig('Figures_B5/trainErr1.png')
##
##
##  plt.figure(2)
##  plt.plot(range(no_epochs_1), te_acc_1)
##  plt.xlabel('Iterations')
##  plt.ylabel('Accuracy')
##  plt.title('Accuracy vs Iterations')
##  plt.savefig('Figures_B5/test_acc1.png')

##  plt.figure(3)
##  plt.plot(range(no_epochs_2), tr_err_2)
##  plt.xlabel('Iterations')
##  plt.ylabel('Error')
##  plt.title('Training error vs Iterations')
##  plt.savefig('Figures_B5/trainErr2.png')
##
##
##  plt.figure(4)
##  plt.plot(range(no_epochs_2), te_acc_2)
##  plt.xlabel('Iterations')
##  plt.ylabel('Accuracy')
##  plt.title('Accuracy vs Iterations')
##  plt.savefig('Figures_B5/test_acc2.png')
##
##  plt.figure(5)
##  plt.plot(range(no_epochs_3), tr_err_3)
##  plt.xlabel('Iterations')
##  plt.ylabel('Error')
##  plt.title('Training error vs Iterations')
##  plt.savefig('Figures_B5/trainErr3.png')
##
##
##  plt.figure(6)
##  plt.plot(range(no_epochs_3), te_acc_3)
##  plt.xlabel('Iterations')
##  plt.ylabel('Accuracy')
##  plt.title('Accuracy vs Iterations')
##  plt.savefig('Figures_B5/test_acc3.png')
##
##      ## Print
##  plt.figure(7)
##  plt.plot(range(no_epochs_4), tr_err_4)
##  plt.xlabel('Iterations')
##  plt.ylabel('Error')
##  plt.title('Training error vs Iterations')
##  plt.savefig('Figures_B5/trainErr4.png')
##
##
##  plt.figure(8)
##  plt.plot(range(no_epochs_4), te_acc_4)
##  plt.xlabel('Iterations')
##  plt.ylabel('Accuracy')
##  plt.title('Accuracy vs Iterations')
##  plt.savefig('Figures_B5/test_acc4.png')
  

if __name__ == '__main__':
  print("main")
  main()
