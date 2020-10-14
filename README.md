# Facial-Emotion-Recognition


<!-- ABOUT THE PROJECT -->
## About The Project

A tensorflow/keras implementation of a facial emotion recognition model based on a convolutional neural network architecture and trained on the fer2013 dataset.

### Built With
* Keras
* Tensorflow
* OpenCV


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* python >= 3.7.9
* keras >= 2.4.3
* tensorflow >= 2.3.1
* opencv >= 4.4
* sklearn >= 0.23
* numpy >= 1.18.5
* pandas >= 1.1.2
* matplotlib >= 3.3.1

### Installation
1. Clone the repo
```sh
git clone https://github.com/MahmoudSabra1/Facial-emotion-recognition
```
2. Install required packages
  * Use [anaconda](https://www.anaconda.com/) to easily install keras and tensorflow in addition to necessary cuda drivers to run the model on GPU.
  ```sh
  conda install tensorflow
  conda install keras
  ```
  * Other packages can be easily installed using either pip or conda.
  ```sh
  pip install opencv
  conda install numpy
  ```
3. Todo: after refactoring of code.
  -open live video capture
  -load video to model
  -train model on new data with other shapes
  -predict on external img


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

<!-- Improving Model Performance -->
## Improving Model Performance

### Baseline Model
Used [neha01 model](https://github.com/neha01/Realtime-Emotion-Detection) as baseline model which is based on a 3 block convolutional neural network architecture. It achieved 66% test accuracy on fer2013 dataset.

### Data Cleaning
Because of alot of mislabeled images in fer2013 dataset, we found that using ferPlus' labels is a better option to train the model for a better performance.

### Regularization
#### 1. Data Augmentation
Data augmentation is used to artifically create images, these images are added to the original training images to increase the total training set size. We implemented data augmentation with keras [ImageDataGenerator](https://keras.io/api/preprocessing/image/#imagedatagenerator-class) class and tuned its parameters. By doing so, we were able to raise the test accuracy by --%. The trick was not to overuse it so that the model could still learn from the training images.

#### 2. Batch Normalization and Dropout Layers
Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1 which makes training faster and more stable.
Dropout layers randomly chooses percentage of input neurons to drop while training such that it has a regularization effect.
Both layers are added to our model increasing performance by --%

<!-- Performance Analysis -->
## Performance Analysis
Plotting the accuracy and loss of the trained model is always the first step to anaylze how the the model is performing. Here are two pictures illustrating the difference in performance between one of the initial architectures used and the final architecture.
--->insert 2 pictures
However, depending on only the accuracy and loss of the trained model doesn't always give a full understanding of the model's performance. There are more advanced metrics that can be used like the F1 score which we used. The F1 score is calculated using two pre-calculated metrics: precision and recall which are best visualised using the confusion matrix. You can checkout (https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd) for a full and clear explanation. Since we designed our model to recognise the 7 universal facial emotions and the ferplus dataset had an 8th class for 'contempt' emotions, we decided to add all contempt class' examples to the 'neutral' class rather than throwing this data away. Here's how our confusion matrix for the 7 classes looks like. F1 score = 0.8
![confusion_matrix](https://user-images.githubusercontent.com/43937873/96011743-9a582e00-0e43-11eb-9b95-eba91f99aa6f.png)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
- https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
- https://machinelearningmastery.com/improve-deep-learning-performance/
- https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
- https://medium.com/analytics-vidhya/deep-learning/home
