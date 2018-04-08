# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 A video recording of my vehicle driving autonomously around the track .
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the following command 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is similar to NVIDIA architecture with slight changes. The data is normalized in the model using a Keras lambda layer and the input is cropped using the Keras Cropping2D layer.

The model includes RELU layers to introduce nonlinearity and a linear activation layer so as to avoid the convergence of the network.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after the first three fully connected layer in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting. 70% of the data was used as training dataset while 30% for the validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I have used the training data provided by the Udacity. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model with single layer convolution followed by fully connected layer I thought this model might be appropriate as few of the discussion on forum was that they were successfully able to do this project with this approach but for me the differnce between the training and the validadtion loss was drastic .It was failing as my car was just moving at an angle of 25 on the track. So after several tries I used NVIDIA architecture mentioned in the udacity classroom. As it was quite helpful. I modified this architecture and used it for training the dataset.

I have included the relu activation layer to introduce nonlinearity. First I tried adding relu activation only after the each convolution layer but the required result was not appropriate and the car was going off track. So I tried adding it after each layer in the network which improved the performance of the car but still it was going offtrack on curves. So I though since the the relu activation makes the network converge much faster so I can replace few relu activation function with other activation function, therefore i used linear activation function for the last layer . This further improved the performance of the model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model by adding dropout layers after the fullyconnected layers.

Then I plotted a graph to vitualize the difference beween these loss.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior I tweaked the dropout values.\

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...
Layer 1: Conv layer with 24 5x5 filters and 2x2 strid, followed by RELU activation
Layer 2: Conv layer with 36 5x5 filters and 2x2 strid, followed by RELU activation
Layer 3: Conv layer with 48 5x5 filters and 2x2 strid, followed by RELU activation
Layer 4: Conv layer with 64 3x3 filters and 1x1 strid, followed by RELU activation
Layer 5: Conv layer with 64 3x3 filters and 1x1 strid, followed by RELU activation
Layer 6: Fully connected layer with 100 neurons, Dropout(0.4) and RELU activation
Layer 7: Fully connected layer with 50 neurons, Dropout(0.3) and RELU activation
Layer 8: Fully connected layer with 10 neurons, Dropout(0.2) and LINEAR activation

#### 3. Creation of the Training Set & Training Process

I used the data provided by the Udacity.

To create the data set out of those data firstly i randomly selected left right and center image corresponding to each steering angle. And depending upon the type of image selected i have modified the steering angles since the left and right cameras point straight, along the length of the car. So the left and right camera are like parallel transformations of the car.
For the left selected image I have added an offset of 0.25 to the steering angle and for the right image i have  subtracted an offset value of 0.25. 
After this I had around 8036 images and steering angles as data set.

To augment the data sat, I also flipped images and angles with a probality 0f 50% thinking that this would be A effective technique for helping with the left turn bias and taking the opposite sign of the steering measurement.

example:

for image,measurement in zip(images,measurements):
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        augment_images.append(cv2.flip(image,1))
        augment_measurement.append(measurement*-1)
    augment_images.append(image)
    augment_measurement.append(measurement)

After the collection process, I had 12006 number of data points. I then preprocessed this data by adding a lambda layer in model as it is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py.

Then In order to train the image faster i have cropped the images from top and bottom by randomply putting a pixel value of 50,20 by using the Cropping2D function provided by the Keras. This is relatively fast, because the model is parallelized on the GPU, so many images are cropped simultaneously.

I finally randomly shuffled the data set and put 30% of the data into a validation set and 70% for training set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the line graph that shows the training and validation loss after each epoch I used an adam optimizer so that manually training the learning rate wasn't necessary.
