# Deep Water surfer detection algorithm

A surfer detection algorithm using a sliding window convolutional neural network implementation inspired by the LeNet-5 network. Written in pure Python.

# Example output video
https://user-images.githubusercontent.com/47697889/110666153-f8a1f680-81c0-11eb-8c2a-9c0411f5af63.mp4

# Example output png
<img width="672" alt="output_example" src="https://user-images.githubusercontent.com/47697889/110666284-14a59800-81c1-11eb-9b3e-32579e49cde6.png">


This repo houses 3 primary scripts which...
- segment images to produce training data
- train a model
- run the model on a webcam video and visualise the results.

Package requirements are listed in the environment file. You can generate an anaconda environment from this by simply running the following in your terminal.

```
conda env create -f environment.yml
```

- Use the config.py file to specify paths to data, parameters, and file destinations 
- Then run the scripts either in an IDE of your choice or directly from the terminal
