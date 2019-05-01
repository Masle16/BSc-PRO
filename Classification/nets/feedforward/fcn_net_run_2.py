from keras.layers import Dense, Activation, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam

def fcn_net():
    model = Sequential()
    model.add(Flatten(input_shape=(224,224,3)))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                        optimizer=Adam(lr=1e-4),
                        metrics=['accuracy'])
    return model
