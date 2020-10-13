# Facial-Emotion-Recognition

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Improving Model Performance](#improving-model-performance)
  * [Baseline Model](#baseline-model)
  * [Data Augmentation](#data-augmentation)
  * [Random Regularization](#random-regularization)
  * [Hyperparameters Tuning](#hyperparameters-tuning)
  * [Multiple Models Ensemble](#multiple-models-ensemble)
* [Acknowledgements](#acknowledgements)



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

### Data Augmentation
Todo

### Random Regularization
Todo

### Hyperparameters Tuning
Todo

### Multiple Models Ensemble
Todo


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
