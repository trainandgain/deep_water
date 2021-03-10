import os

import numpy as np 
import cv2

import config


def init_img(import_dir):
    return cv2.imread(import_dir, cv2.IMREAD_GRAYSCALE)

def sliding_window(image, step_size, window_size):
    'Slide a window across the image and yield segments produced'
    for y in range(0, image.shape[0], step_size): 
        for x in range(0, image.shape[1], step_size):
            segment = image[y:y + window_size[1], x:x + window_size[0]]
            seg_size = np.size(segment)
            # Check if segment is full-size, (not on edge of img)
            if seg_size == (window_size[0] * window_size[1]):
                yield segment

def export_segments(segments, export_dir):
    'Export segments created to export_dir'
    count = 0 
    for seg in segments:
        count += 1
        cv2.imwrite(export_dir + 'seg' + str(count) + '.jpg', seg)
        
def get_sliding_window_segments(import_dir, export_dir,
                                step_size, window_size):
    'Generate segments from images in import_dir and write to export_dir'
    
    for dirname, _, filenames in os.walk(import_dir):
        # Start count
        count = 0
        for filename in filenames:
            count += 1
            # Find file path for image
            import_file_path = os.path.join(dirname, filename)
            # Import image
            img = init_img(import_file_path)
            # Segment image using a sliding window
            segments = sliding_window(img, step_size, window_size)
            # Export the segments
            export_segments(segments, export_dir + '/img' + str(count))
            # Print update
            print('Image ' + str(count) + ' segmented!')


if __name__ == '__main__':
    
    # From config
    import_file = config.FULL_IMAGES
    export_file = config.IMAGE_SEGMENTS
    step_size = config.STEP_SIZE
    window_size = config.WINDOW_SIZE

    # Run
    get_sliding_window_segments(import_file, export_file, step_size, window_size)
    