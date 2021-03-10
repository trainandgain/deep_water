import numpy as np 
import cv2
import matplotlib.pyplot as plt

from tensorflow import keras

from img_utils import non_max_suppression

import config


def init_img(import_file_path):
    return cv2.imread(import_file_path, cv2.IMREAD_GRAYSCALE)

def sliding_window(image, step_size, window_size):
    ''' Slide a window across the image and yield the individual
    segments and corner coodinates
    '''
    seg_shape = (1,) + window_size + (1,)  # Additional dim required for keras
    coord_shape = (1, 4)
    segs = np.empty(seg_shape)
    coords = np.empty(coord_shape)
    
    for y in range(370, image.shape[0], step_size): 
        for x in range(0, (image.shape[1] - 200), step_size):
            segment = image[y:y + window_size[1], x:x + window_size[0]]
            seg_size = np.size(segment)
            
            # Check if segment is full-size (not on edge of image)
            if seg_size == (window_size[0] * window_size[1]):
                coord = np.array([x, y, x + window_size[0], y + window_size[1]])
                
                # Append segment to segs; coord to coords
                segment_reshaped = np.reshape(segment, seg_shape)
                coord_reshaped = np.reshape(coord, coord_shape)
                segs = np.concatenate((segs, segment_reshaped))
                coords = np.concatenate((coords, coord_reshaped)) 
                
    # Delete redundant first row created on array init
    segs = np.delete(segs, 0, 0)
    coords = np.delete(coords, 0, 0)
    
    return segs, coords

def get_predictions(X, segment_coords, model):
    'Returns model predictions and segment coords in single array'
    predictions = model.predict(X)    
    coords_preds = np.concatenate((segment_coords, predictions), axis=1)
    return coords_preds

def select_picks(coords_preds, 
                             proba_threshold=0.5, 
                             overlap_threshold=0.01
                             ):
    'Process model output to get selections of surfer location in image'
    # Drop predictions below 0.5 probability threshold
    coords_preds = coords_preds[coords_preds[:, 4] >= proba_threshold]
    
    # Seperate bounding box coordinates and corresponding probabilities
    boxes = coords_preds[:, :4]
    preds = coords_preds[:, 4]

    # Apply Adrian Rosebrock's NMS function
    picks = non_max_suppression(boxes, preds, overlap_threshold)

    return picks
    
def visualise(img, picks, save_file=None):
    'Overlay picks and surfer count over image'
    # Count the remaining bounding boxes for surfer numbers
    count = picks.shape[0]
    
    # Grab corner coordinates of the bounding boxes
    x1 = picks[:, 0]
    y1 = picks[:, 1]
    x2 = picks[:, 2]
    y2 = picks[:, 3]

    # Place rectangular markers around surfers
    for i in range(0, count):
        # Width and height of box
        w = x2[i] - x1[i]
        h = y2[i] - y1[i]
        # Draw rectangle markers
        cv2.rectangle(img, 
                      (x1[i], y1[i]), 
                      ((x1[i]+w), (y1[i]+h)), 
                      (255,0,0), 
                      2)

    # Show surfer numbers on plot
    plt.text(1150, 650, str(count), fontsize=20)

    #if save_file:
    #    plt.imsave(save_file, np.array(img), cmap='gray')
    #else:
        # Show image
    plt.imshow(np.array(img), cmap='gray')

def run_single_img(img, model):
    'Visualise model result on jpeg file'
    
    # Segment image using sliding window
    segments, coords = sliding_window(img, step_size=8, window_size=(22, 22))
    
    # Feature scale the input pixel values for model
    X = segments / 255
    
    # Load and run model
    coords_preds = get_predictions(X, coords, model)
    
    # Select picks for surfer locations
    picks = select_picks(coords_preds)
    
    visualise(img, picks, save_file)
    
def run_video(folder):
    'Visualise model result on folder of jpeg files'
    # read in video frame by frame
    
    
    # for frame in frames
        # run_single_img    
        # write frame to video out
    
    # end




if __name__ == '__main__':
    
    # Image file
    single_img = 'inputs/frame0.jpg'
    save_file = 'outputs/example.jpg'
    
    # Import trained model
    model = keras.models.load_model(config.TRAINED_MODEL)
    
    # Import image
    img = init_img(img_file)
    
    run_single_img(single_img, model)
    