# Deep Water surfer detection algorithm

A surfer detection algorithm using a sliding window convolutional neural network implementation inspired by the LeNet-5 network. Written in pure Python.

# Example output video
https://user-images.githubusercontent.com/47697889/110752203-d77eeb80-823c-11eb-8d5f-cfa887c8829b.mp4


# Example output png
<img width="672" alt="output_example" src="https://user-images.githubusercontent.com/47697889/110666284-14a59800-81c1-11eb-9b3e-32579e49cde6.png">

# How to use this repository
This repo houses 3 primary scripts which...
- segment images to produce training data
- train a model
- run the model on a webcam video and visualise the results.

Package requirements are listed in the environment file. You can generate an anaconda environment from this by navigating to the 'deep_water' root directory and running the following in your terminal.

```
conda env create -f environment.yml
```

## segment_images.py
...splits images in a given folder into segments of size 'window_size' and steps through the image with step size defined by 'step_size'. The purpose of this script is primarily to generate data ready for labelling. Once these sements are split into positive/negative examples, then you are ready to train your model. Use the config.py file to specify the images folder, window size, step size, and file destinations.

## train_model.py
...by default trains a small LeNet-5 style conv net to classify image segments as 1, if they include a surfer, or 0 if they do not. Set paths to folders of positive and negative training image segments in the config, and you are good to go. You can also set training parameters such as number of epochs, batch size, and validation split here.

## inference.py
...visualises the output of your model on an mp4 video input. It takes in every other frame, processes it so it can be used as model input, then applies non-maximal suppression to the models picks, and plots bounding boxes and a surfer number count in the output video. Set the location of your trained model, mp4 video input and output, and a frame limit (if you are inputting a large video) and then press play. 
