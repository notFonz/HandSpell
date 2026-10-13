# Datasets

## A. Static Cropped Labeled Images

### 1. Sign Language MNIST - Preprocessed ([link](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?select=sign_mnist_test))

- 28x28 grey-scaled images (1 channel) already flattened
- 24 classes of letters (excluding J and Z which require motion)
- 27,455 + 7,172 images (pre-split into train and test)
- result of applying a pre-processing pipeline to 1,704 color images (3 channels) - based on ImageMagick:
  - crop hands only
  - grey-scale
  - resize
  - data augmentation
    - filters ('Mitchell', 'Robidoux', 'Catrom', 'Spline', 'Hermite')
    - 5% random pixelation
    - +/- 15% brightness/contrast
    - 3 degrees rotation


### 2. Sign Language MNIST - Original ([link](https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/blob/master/Dataset.zip))

- 240x320 color images (3 channels)
- 24 classes of letters (excluding J and Z which require motion)
- 1,704 images from 7 individuals
