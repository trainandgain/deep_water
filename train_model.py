
import os
import h5py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

def init_neural_network(lambd=0.0):
    # Initialise keras sequential model
    model = keras.Sequential()
    
    # Add convolutional network layers
    # Input layer, [3x3 2D convolution, with 3 filters,
    # followed by a 2x2 max pooling layer]
    model.add(tf.keras.layers.Conv2D(3, (3,3), strides=(1,1), padding='valid',
                                     activation = 'relu', input_shape=(22, 22, 1),
                                     kernel_regularizer=keras.regularizers.l2(lambd)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=None, padding='valid'))
    
    # 2nd layer, [3x3 2D convolution, with 6 filters,
    # followed by a 2x2 max pooling layer]
    model.add(tf.keras.layers.Conv2D(6, (3,3), strides=(1,1), padding='valid',
                                     activation = 'relu',
                                     kernel_regularizer=keras.regularizers.l2(lambd)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=None, padding='valid'))
    
    # Flatten to a 1 dimensional layer
    model.add(tf.keras.layers.Flatten())
    # 3rd layer, 22 unit dense connected layer
    model.add(tf.keras.layers.Dense(22, activation='relu'))
    
    # Output layer into a single sigmoid unit
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile model, [using stochastic gradient descent optimiser,
    # and binary crossentropy loss function]
    model.compile('SGD', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def import_training_data(dir_path):
    """Docstring...
    """
    # Initialize numpy array of appropriate dimensions
    examples = np.ndarray((1, 22, 22))
    # Loop through files in training segments directory
    for dirname, _, filenames in os.walk(dir_path):
        for filename in filenames:
            # Import files as numpy arrays
            path = os.path.join(dirname, filename)
            example = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            example_reshaped = np.reshape(example ,(1, 22, 22))
            # Synthesize training examples by laterally inverting images
            example_flipped = cv2.flip(example, 1)
            example_flipped_reshaped = np.reshape(example_flipped ,(1, 22, 22))
            # Concatenate examples into one numpy array
            examples = np.concatenate((examples, example_reshaped), axis=0)
            examples = np.concatenate((examples, example_flipped_reshaped), axis=0)
            
    # Delete the first row of zeros created when examples array was initialised
    examples = np.delete(examples, (0), axis=0)
    
    return examples

def run():
    # Import and format data
    path_pos = '/kaggle/input/positive-seg-examples/training_segments_positive'
    pos_examples = import_training_data(path_pos)

    path_neg = '/kaggle/input/negative-seg-examples/training_segments_negative'
    neg_examples = import_training_data(path_neg)

    # Concatenate datasets
    data = np.concatenate((pos_examples, neg_examples), axis=0)
    X = np.reshape(data, (data.shape[0], 22, 22, 1))

    # Make labels for concatenated image matrix
    labels = np.concatenate((np.ones((pos_examples.shape[0], 1)),
                             np.zeros((neg_examples.shape[0], 1))), axis=0)

    # Feature scale pixel values
    X = data / 255
    # Format image data to be keras readable
    X = np.reshape(X, (X.shape[0], 22, 22, 1))

    # Shuffle data, seeding so it shuffles labels and images together
    np.random.seed(0)
    np.random.shuffle(X)
    np.random.shuffle(labels)

    # Train and save model
    model = init_neural_network(lambd=0)
    model.fit(X, labels, epochs=5, validation_split=0.15)
    model.save('/kaggle/working/test_model.hdf5')

    
    
if __name__ == '__main__':
    
    path_pos = '/kaggle/input/positive-seg-examples/training_segments_positive'
    path_neg = '/kaggle/input/negative-seg-examples/training_segments_negative'
    
    epochs = 5
    val_split = 0.15
    
    run()
    
    