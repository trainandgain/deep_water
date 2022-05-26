import os
import sys
import cv2
import numpy as np

def main(path, video_name):
    """
    Main function,
    Input: path to video, video_name
    Splits video into frames... 
    """
    # name of video
    name = os.path.splitext(video_name)[0]
    # out path
    out_folder = os.path.join('out/', name) 
    os.makedirs(out_folder, exist_ok=False)
    # load video
    cap = cv2.VideoCapture(path)
    # split vidoes
    frame_N = 0
    # open cap
    print('Splitting frames...')
    while(cap.isOpened()):
        # returned and frame
        ret, frame = cap.read()
        # only save if True
        if ret:
            cv2.imwrite(f"{out_folder}/{frame_N}.png", frame)
        else:
            break
        # frame number
        frame_N += 1
    print('All done...')
    # sort out cv2 releases
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # get video to load
    base_folder = 'video/'
    video_name = sys.argv[1]
    full_path = os.path.join(base_folder, video_name)
    assert os.path.exists(full_path)
    main(full_path, video_name)