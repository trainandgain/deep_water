# Image segmentation config

FULL_IMAGES = 'inputs/images...'

IMAGE_SEGMENTS = 'inputs/segments...'



# Training config

MODEL_PATH = 'models/new_model.hdf5'

POSITIVE_TRAINING_SEGS = 'inputs/training_segments_positive'

NEGATIVE_TRAINING_SEGS = 'inputs/training_segments_negative'

STEP_SIZE = 8 

WINDOW_SIZE = [22, 22]

EPOCHS = 50

BATCH_SIZE = 32

VAL_SPLIT = 0.10

RANDOM_STATE = 42



# Inference config

TRAINED_MODEL = 'models/LeNet5_trained_model.hdf5'

MP4_FILE = 'inputs/magicseaweed.croyd.mp4'

VIDEO_OUT = 'outputs/test_06.avi'
    
FRAME_LIMIT = 480
