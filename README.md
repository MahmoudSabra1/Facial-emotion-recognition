# Facial-Emotion-Recognition


<!-- ABOUT THE PROJECT -->
## About The Project

A tensorflow implementation of facial emotion recognition model trained on fer2013 dataset.

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

_For more examples, please refer to the [Documentation](https://example.com)_


<!-- Improving Model Performance -->
## Improving Model Performance
### Baseline Model
Used [neha01 model](https://github.com/neha01/Realtime-Emotion-Detection) as baseline model which achieved 66% test accuracy on fer2013 dataset.

### Cleaning Data
Because of alot of mislabeled images in fer2013 dataset, we found that ferPlus labels is a better option to train the model that led to improved test accuracy.

### Regularization
#### 1. Data Augmentation
Implemented it with keras [ImageDataGenerator](https://keras.io/api/preprocessing/image/#imagedatagenerator-class) class and tuned its parameters so the test accuracy increased by --%.

#### 2. Batch Normalization and Dropout Layers
Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1 which makes training faster and more stable.
Dropout layers randomly chooses percentage of input neurons to drop while training such that it has a regularization effect.
Both layers are added to our model increasing performance by --%


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
- https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
- https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
