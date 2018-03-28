import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from model import model

def load_images_from_folder(folder):
    images = []
    file_names = []
    for filename in os.listdir(folder):
        img = scipy.misc.imresize(cv2.imread(os.path.join(folder, filename)), 0.3)
        if img is not None:
            images.append(img)
            file_names.append(filename)

    return images, file_names

def simple_augment(images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq.augment_images(images)

    return images_aug

def heavy_augment(images):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [
                           # Convert some images into their superpixel representation,
                           # sample between 20 and 200 superpixels per image, but do
                           # not replace all superpixels with their average, only
                           # some of them (p_replace).
                           sometimes(
                               iaa.Superpixels(
                                   p_replace=(0, 1.0),
                                   n_segments=(20, 200)
                               )
                           ),

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's chanell with 5% probability.
                           # This sets each pixel value v to 255-v.
                           iaa.Invert(0.05, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           # In some images move pixels locally around (with random
                           # strengths).
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                           # In some images distort local areas with varying strength.
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    images_aug = seq.augment_images(images)

    return images_aug

def load_data():
    start_time = time.time()
    # Load input(X)
    images, file_names = load_images_from_folder('pokemon_images')
    images_simple_aug = simple_augment(images)
    images_heavy_aug = heavy_augment(images)

    image_data = np.asarray(images + images_simple_aug + images_heavy_aug)
    # Normalize image vectors
    image_data_flatten = image_data.reshape(image_data.shape[0], -1).T / 255

    num_example = len(image_data)
    num_element_x = images[0].shape[0] * images[0].shape[1] * images[0].shape[2]

    print "Input size: {}".format(image_data_flatten.shape)
    assert image_data_flatten.shape == (num_element_x, num_example)

    # Load output(Y)
    labels = [int(filename.strip('0').strip('.png')) for filename in file_names]
    label_data = labels + labels + labels

    num_element_y = len(labels)
    labels_flatten = np.zeros((num_element_y, num_example))

    for i in range(0, num_example - 1):
        labels_flatten[label_data[i] - 1][i] = 1

    print "Output size: {}".format(labels_flatten.shape)
    assert labels_flatten.shape == (num_element_y, num_example)

    # Shuffle data to train, test
    X_train, X_test, y_train, y_test = train_test_split(image_data_flatten.T, labels_flatten.T, test_size=0.2, random_state=42)

    # Shuffle data to train, dev
    # X_train, X_test, X_dev, X_dev_test = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

    print "Time used {} second for loading data..".format(time.time() - start_time)

    return X_train.T, X_test.T, y_train.T, y_test.T

if __name__ == "__main__":
    start_time = time.time()

    X_train, X_test, y_train, y_test = load_data()

    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(y_test.shape))

    # Training model
    # layers_dims = (X_train.shape[0], 100, y_train.shape[0])
    parameters = model(X_train, y_train, X_test, y_test, learning_rate = 0.0001, num_epochs = 100, minibatch_size = 32, print_cost = True)
    # parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.075, num_iterations=10, print_cost=True)
    # print "Training finished! Time used {} seconds!".format(time.time() - start_time)

    # Predict
    # predictions = predict(parameters, X_test)

    # print predictions
    # print y_test
    # print predictions - y_test
    # print (np.sum(predictions - y_test, axis=0) == 0).shape

    # print "Accuracy: {}".format(np.sum(np.sum(predictions - y_test, axis=0) == 0)/y_test.shape[1])
    # print ('Accuracy: %d' % float((np.dot(y_test, predictions.T) + np.dot(1 - y_test, 1 - predictions.T)) / float(y_test.size) * 100) + '%')
