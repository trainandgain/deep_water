import numpy as np 
import cv2

from tensorflow import keras

from non_max_supp import non_max_suppression

import config


def init_img(import_file_path):
    return cv2.imread(import_file_path, cv2.IMREAD_GRAYSCALE)

def sliding_window(image, step_size, window_size):
    'Slide a window across image and yield image segments and corner coords'
    seg_shape = (1,) + window_size + (1,)  # Additional dim required for keras
    coord_shape = (1, 4)
    segs = np.empty(seg_shape)
    coords = np.empty(coord_shape)
    
    for y in range(370, image.shape[0], step_size): 
        for x in range(0, (image.shape[1] - 250), step_size):  # FROM 200
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
                             proba_threshold=0.8, 
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
    
def run_video(mp4_file, video_out, model, frame_limit):
    'Returns model overlay on mp4 video input'
    # Read in video frame by frame
    cap = cv2.VideoCapture(mp4_file)
    
    if cap.isOpened() == False:
        raise ValueError('Error opening mp4 file')
    
    vid_resolution = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video encoder for Windows10
    out = cv2.VideoWriter(video_out, fourcc, 12, vid_resolution)
    
    # Count frames processed
    frame = 0
    
    while cap.isOpened() and frame <= frame_limit:
        ret, rgb_img = cap.read()
        if ret == True:
            # Only process every other frame
            frame += 1
            if frame % 30 == 0:
                continue
            
            # Convert to black and white for processing
            bw_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            # Get model picks
            _, picks = run_single_img(bw_img, model)
            # Visualise results
            output = plot_bounding_boxes(rgb_img, picks)
            # Write to jpeg
            cv2.imwrite(f'outputs/out_{frame}.jpg', output)
            # Write to video
            out.write(output)
            
            print(f'frame {frame} processed!')
               
        # When end of video reached break loop
        else:
            break
    
    # Release video capture and output objects
    cap.release()
    out.release() 




if __name__ == '__main__':
    
    # File paths
    mp4 = config.MP4_FILE
    video_out = config.VIDEO_OUT
    frame_limit = config.FRAME_LIMIT
    model_file = config.TRAINED_MODEL
    
    # Import trained model
    model = keras.models.load_model(model_file)

    # Run on mp4 video    
    run_video(mp4, video_out, model, frame_limit)
    