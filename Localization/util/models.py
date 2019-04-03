from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np


def cnn_net(out_size=8):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))    # 32
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))                               # 64
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  #  3D activation map -> 1D vector
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3, epsilon=10e-8),
                  metrics=['accuracy'])
    return model


def load_model(name):
    
    # open json file with model
    json_file = open('saved_models/'+name+'.json', 'r')
    model_json = json_file.read()
    json_file.close()
    # create model from json
    model = model_from_json(model_json)
    # load weights into model
    model.load_weights('saved_models/'+name+'.h5')

    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, epsilon=10e-8, decay=0.025),
                  metrics=['accuracy'])
    
    return model


def measure_accuracy(predicted, ground_truth):
    
    acc = np.mean(np.equal(np.argmax(predicted,axis=-1),np.argmax(ground_truth,axis=-1)))
    
    return acc
