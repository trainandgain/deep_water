import os

import numpy as np
import cv2
from tensorflow import keras

import config

def init_neural_network(lambd=0.0):
    
    # Initialise keras sequential model
    model = keras.Sequential()
    
    # Add convolutional network layers
    
    # Input layer, [3x3 2D convolution, with 3 filters,
    # followed by a 2x2 max pooling layer]
    model.add(keras.layers.Conv2D(3, (3,3), strides=(1,1), padding='valid',
                                     activation = 'relu', input_shape=(22, 22, 1),
                                     kernel_regularizer=keras.regularizers.l2(lambd)))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=None, padding='valid'))
    
    # 2nd layer, [3x3 2D convolution, with 6 filters,
    # followed by a 2x2 max pooling layer]
    model.add(keras.layers.Conv2D(6, (3,3), strides=(1,1), padding='valid',
                                     activation = 'relu',
                                     kernel_regularizer=keras.regularizers.l2(lambd)))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=None, padding='valid'))
    
    # Flatten to a 1 dimensional layer
    model.add(keras.layers.Flatten())
    
    # 3rd layer, 22 unit dense connected layer
    model.add(keras.layers.Dense(22, activation='relu'))
    
    # Output layer into a single sigmoid unit
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile model using stochastic gradient descent optimiser,
    # and binary cross-entropy loss function
    model.compile('SGD', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def import_training_data(dir_path):
    """Docstring...
    """
    # Initialize numpy array of appropriate dimensions
    examples = np.ndarray((1, 22, 22))
    
    # Loop through image files in training segments directory
    for _, _, filenames in os.walk(dir_path):
        for filename in filenames:
            
            # Import files as numpy arrays
            path = os.path.join(dir_path, filename)
            example = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            example_reshaped = np.reshape(example ,(1, 22, 22))
            
            # Synthesize training examples by laterally inverting images
            example_flipped = cv2.flip(example, 1)
            example_flipped_reshaped = np.reshape(example_flipped ,(1, 22, 22))
            
            # Concatenate original and transformed example into single array
            examples = np.concatenate((examples, example_reshaped))
            examples = np.concatenate((examples, example_flipped_reshaped))
            
    # Delete the first row of zeros created when examples array was initialised
    examples = np.delete(examples, (0), axis=0)
    
    return examples

def run(path_pos, path_neg, epochs, val_split, model_path=None):
    ''
    
    # Import and format data
    pos_examples = import_training_data(path_pos)
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
    
    # Train model
    model = init_neural_network(lambd=0.0)
    model.fit(X, 
              labels, 
              epochs=epochs, 
              validation_split=val_split,
              batch_size=1,
              shuffle=True)
    
    # Save trained model to specified location
    if model_path:
        model.save(model_path)
    
    return model


if __name__ == '__main__':
    
    path_pos = config.POSITIVE_TRAINING_SEGS
    path_neg = config.NEGATIVE_TRAINING_SEGS
    epochs = config.EPOCHS
    val_split = config.VAL_SPLIT
    random_state = config.RANDOM_STATE
    model_path = config.MODEL_PATH
    
    run(path_pos, path_neg, epochs, val_split, model_path)
    