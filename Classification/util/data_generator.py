# Keras imports
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
# General imports
from os import listdir
# Own .py functions
import util.image_import as ii

def make_data_generator(train_path, test_path, val_path="", load_ram=False, ignore=[], augmentation=True, preprocessing=[True, True], mobilenet=False):

    
    mean_image_train = ii.calulate_mean(train_path, ignore) # Calculates mean for each channel for every pixel
    def subtract_mean(img):
        return img - mean_image_train
    
    batch_size = 32
    # Calculate classes 
    list_sub_dir = listdir(train_path)
    # Removes folder which is in ignore
    for i in ignore:
        list_sub_dir.remove(i)
    num_classes = len(list_sub_dir)    
    
    # Dataugmentation and preprocessing variables initilization
    rescale=None
    rotation_range=0
    vertical_flip=False
    horizontal_flip=False
    brightness_range=None
    preprocessing_function=None
       
    if augmentation: # sets varibles for augmentation
        rotation_range=10
        vertical_flip=True
        horizontal_flip=True
        brightness_range=(0.65, 1.35)
   
    if preprocessing[0]: # Rescale preprocessing
        rescale=1./255
    if preprocessing[1]: # subtract mean preprocessing function
        if mobilenet:
            preprocessing_function=keras.applications.mobilenet.preprocess_input
        else:
            preprocessing_function=subtract_mean
    
    # Declare different datagenerator with parameter from above
    train_datagen = ImageDataGenerator(
    rescale=rescale,
    rotation_range=rotation_range,
    vertical_flip=vertical_flip,
    horizontal_flip=horizontal_flip,
    brightness_range=brightness_range,
    preprocessing_function=preprocessing_function)
    # Test
    test_datagen = ImageDataGenerator(
    rescale=rescale,
    preprocessing_function=preprocessing_function)
   
    if val_path == "":
        if not load_ram:
            # Train data generator
            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(224, 224),
                classes=list_sub_dir, # Classes defined by directories
                batch_size=batch_size, shuffle=True)

            # Test data generator
            test_generator = test_datagen.flow_from_directory(
                test_path,
                target_size=(224, 224),
                classes=list_sub_dir, # Classes defined by directories
                batch_size=batch_size, shuffle=True)
            
            return train_generator, test_generator
        elif load_ram:
            X_train, y_train = ii.images_to_numpy_full_class(train_path, ignore)
            X_test, y_test = ii.images_to_numpy_full_class(test_path, ignore)

            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)

            train_generator = train_datagen.flow(X_train,
                                                 y_train,
                                                 batch_size=batch_size, shuffle=True)

            test_generator = test_datagen.flow(X_test,
                                                 y_test,
                                                 batch_size=batch_size, shuffle=True)

            print("Found Training " + str(X_train.shape[0]) + " images belonging to " + str(num_classes) + " classes")
            print("Found Test " + str(X_test.shape[0]) + " images belonging to " + str(num_classes) + " classes")
            
            return train_generator, test_generator        
    else:
        if not load_ram:
            
            # Train data generator
            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(224, 224),
                classes=list_sub_dir, # Classes defined by directories
                batch_size=batch_size)
            # Valid data generator
            valid_generator = valid_datagen.flow_from_directory(
                val_path,
                target_size=(224, 224),
                classes=list_sub_dir, # Classes defined by directories
                batch_size=batch_size)
            # Test data generator
            test_generator = valid_datagen.flow_from_directory(
                test_path,
                target_size=(224, 224),
                classes=list_sub_dir, # Classes defined by directories
                batch_size=batch_size)
            
            return train_generator, valid_generator, test_generator
        elif load_ram:
            X_train, y_train = ii.images_to_numpy_full_class(train_path, ignore)
            X_valid, y_valid = ii.images_to_numpy_full_class(val_path, ignore)
            X_test, y_test = ii.images_to_numpy_full_class(test_path, ignore)

            y_train = to_categorical(y_train, num_classes)
            y_valid = to_categorical(y_valid, num_classes)
            y_test = to_categorical(y_test, num_classes)

            
            train_generator = train_datagen.flow(X_train,
                                                 y_train,
                                                 batch_size=batch_size, shuffle=True)
            
            valid_generator = valid_datagen.flow(X_valid,
                                                 y_valid,
                                                 batch_size=batch_size, shuffle=True)
            
            test_generator = valid_datagen.flow(X_test,
                                                 y_test,
                                                 batch_size=batch_size, shuffle=True)

            print("Found Training " + str(X_train.shape[0]) + " images belonging to " + str(num_classes) + " classes")
            print("Found Validation " + str(X_valid.shape[0]) + " images belonging to " + str(num_classes) + " classes")
            print("Found Test " + str(X_test.shape[0]) + " images belonging to " + str(num_classes) + " classes")
            return train_generator, valid_generator, test_generator        
