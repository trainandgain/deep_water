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
    
def plot_bounding_boxes(img, picks):
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
                      (255,255,255), 
                      2)
        
    # Show surfer numbers on plot
    #plt.text(1150, 650, str(count), fontsize=20)
    cv2.putText(img, str(count), (1150, 650), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, 
                (255,255,255), 3)
    
    return img
    #if save_file:
    #    plt.imsave(save_file, np.array(img), cmap='gray')
    #else:
    # Show image
    #cv2.imshow('frame', np.array(img))#, cmap='gray')

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
    
    output_img = plot_bounding_boxes(img, picks)
    
    return output_img, picks
    
def run_video(mp4_file, model):
    'Visualise model result on folder of jpeg files'
    # Read in video frame by frame
    cap = cv2.VideoCapture(mp4_file)
    
    if cap.isOpened() == False:
        raise ValueError('Error opening mp4 file')
    
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))
    
    frame = 0
    
    while cap.isOpened():   #and frames_processed <= n_frames:
        ret, rgb_img = cap.read()
        if ret == True:
            # Count frames processed
            frame += 1
            # Convert to black and white for processing
            bw_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            # Get model picks
            _, picks = run_single_img(bw_img, model)
            # Visualise results
            output = plot_bounding_boxes(rgb_img, picks)
            # Write to img
            cv2.imwrite(f'output/out_{frame}.jpg', output)
            print(f'frame {frame} processed!')
            
            # TODO: write to video
            # Display the resulting frame
            #cv2.imshow('Frame', output)
            # Press Q on keyboard to exit
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    break

        # Break the loop
        else:
            break
     
    # When everything done, release the video capture object
    cap.release()
    out.release() 
    
    # Closes all the frames
    cv2.destroyAllWindows()
    
    # end
    



if __name__ == '__main__':
    
    # File paths
    single_img = 'inputs/frame0.jpg'
    mp4 = 'inputs/magicseaweed.croyd.mp4'
    
    # Import trained model
    model = keras.models.load_model(config.TRAINED_MODEL)
    
    # Import image
    #img = init_img(single_img)
    #output, _ = run_single_img(img, model)

    # Run on mp4 video    
    run_video(mp4, model)
    
    