import os

import numpy as np 
import cv2
import h5py
import matplotlib.pyplot as plt

from tensorflow import keras

from img_utils import non_max_suppression

def init_img(import_file_path):
    return cv2.imread(import_file_path, cv2.IMREAD_GRAYSCALE)

def sliding_window(image, step_size, window_size):
    ''' Slide a window across the image and yield the individual
    segments.
    '''
    for y in range(370, image.shape[0], step_size): 
        for x in range(0, (image.shape[1] - 200), step_size):
            segment = image[y:y + window_size[1], x:x + window_size[0]]
            seg_size = np.size(segment)
            # Check if segment is full-size, (not on the edge of image)
            if seg_size == (window_size[0] * window_size[1]):
                yield segment

#TODO: Amalgamate sliding_window and sliding_window_coords functions
def sliding_window_coords(image, step_size, window_size):
    ''' Slide a window across the image and yield the coordinates
    for the segments.
    '''
    for y in range(370, image.shape[0], step_size): 
        for x in range(0, (image.shape[1] - 200), step_size):
            segment = image[y:y + window_size[1], x:x + window_size[0]]
            seg_size = np.size(segment)
            # Check if segment is full-size, (not on edge of img)
            if seg_size == (window_size[0] * window_size[1]):
                yield np.array([x, y, x + window_size[0], y + window_size[1]])
                



if __name__ == '__main__':
    
    # Import image file
    import_file_path = '/kaggle/input/single-frame/frame0.jpg'
    img = init_img(import_file_path)

    # Segment image using sliding window
    segments = sliding_window(img, step_size=8, window_size=[22, 22])

    # Wrangle segments into a numpy array to feed to model
    examples = np.ndarray((1, 22, 22, 1))
    for segment in segments:
            reshaped = np.reshape(segment ,(1, 22, 22, 1))
            examples = np.concatenate((examples, reshaped), axis=0)
    examples = np.delete(examples, 0, 0)

    # Fetch sliding window corner coordinates 
    xy_coords = sliding_window_coords(img, step_size=8, window_size=[22, 22])
    # Wrangle xy_coords iterator into numpy array for non-maximal-suppression
    xy_mat = np.ndarray((1, 4))
    for xy_coord in xy_coords:
        reshaped = np.reshape(xy_coord ,(1, 4))
        xy_mat = np.concatenate((xy_mat, reshaped), axis=0)
    xy_mat = np.delete(xy_mat, 0, 0)

    # Featur scale the input pixel values for model
    X = examples / 255

    # Load and run model
    model = keras.models.load_model('/kaggle/input/lenet-model/test_model_LeNet.hdf5')
    model.summary()
    predictions = model.predict(X)
    xy_preds = np.concatenate((xy_mat, predictions), axis=1)

    # Drop predictions below 0.5 probability threshold
    xy_preds = xy_preds[xy_preds[:, 4] >= 0.5]

    # Seperate bounding box coordinates and corresponding probabilities
    boxes = xy_preds[:, :4]
    probs = xy_preds[:, 4]

    # Apply Adrian Rosebrock's NMS function
    picks = non_max_suppression(boxes, probs, overlapThresh=0.01)

    # Count the remaining bounding boxes for surfer numbers
    count = len(picks)

    # Visualise results

    # Grab corner coordinates of the bounding boxes
    x1 = picks[:, 0]
    y1 = picks[:, 1]
    x2 = picks[:, 2]
    y2 = picks[:, 3]

    # Place rectangular markers around surfers
    for i in range(0, len(x1)):
        # Width and height of box
        w = x2[i] - x1[i]
        h = y2[i] - y1[i]
        # Draw rectangle markers
        cv2.rectangle(img,(x1[i], y1[i]), ((x1[i]+w),
                     (y1[i]+h)), (255,0,0), 2)

    # Show surfer numbers on plot
    plt.text(1150, 650, str(count), fontsize=20)

    # Save and show image
    plt.imshow(np.array(img), cmap='gray')
    plt.savefig('proto_output.jpg', dpi=300)
