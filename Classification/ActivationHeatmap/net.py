from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

def cnn_net():
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
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, epsilon=10e-8),
                  metrics=['accuracy'])
    return model
