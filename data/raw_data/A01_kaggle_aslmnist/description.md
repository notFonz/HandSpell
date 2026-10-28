Downloaded zip from [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download)


### Sign Language MNIST ([link](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?select=sign_mnist_test))

Dataset: A01_kaggle_aslmnist
- 34627 images (pre-split into 27,455 train & 7,172 test)
- 24 labels (['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])
- images 28x28 pixels
- All greyscale images (1 channel)

Result of applying a pre-processing pipeline to 1,704 color images, see `B01_original_aslmnist` - based on ImageMagick:
  - crop hands only
  - grey-scale
  - resize
  - data augmentation
    - filters ('Mitchell', 'Robidoux', 'Catrom', 'Spline', 'Hermite')
    - 5% random pixelation
    - +/- 15% brightness/contrast
    - 3 degrees rotation


Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a **single 28x28 pixel image with grayscale values between 0-255**.

The original hand gesture image data represented multiple users repeating the gesture against different backgrounds.

The Sign Language MNIST data came from greatly extending the small number (1704) of the color images included as not cropped around the hand region of interest.

To create new data, **an image pipeline** was used **based on ImageMagick** and included **cropping to hands-only, gray-scaling, resizing, and then creating at least 50+ variations to enlarge the quantity**. The modification and expansion strategy was filters ('Mitchell', 'Robidoux', 'Catrom', 'Spline', 'Hermite'), along with 5% random pixelation, +/- 15% brightness/contrast, and finally 3 degrees rotation.

Because of the tiny size of the images, these modifications effectively alter the resolution and class separation in interesting, controllable ways.
