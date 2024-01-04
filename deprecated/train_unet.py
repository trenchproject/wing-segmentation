from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.metrics import MeanIoU

"""
Multiclass semantic segmentation using U-Net
"""

def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
  # Build the model
  inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  s = inputs

  #Contraction path
  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
  c1 = Dropout(0.1)(c1)
  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
  p1 = MaxPooling2D((2, 2))(c1)

  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
  c2 = Dropout(0.1)(c2)
  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
  p2 = MaxPooling2D((2, 2))(c2)

  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
  p3 = MaxPooling2D((2, 2))(c3)

  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
  p4 = MaxPooling2D(pool_size=(2, 2))(c4)

  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
  c5 = Dropout(0.3)(c5)
  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

  #Expansive path
  u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

  u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

  u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
  c8 = Dropout(0.1)(c8)
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

  u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
  c9 = Dropout(0.1)(c9)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

  outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

  model = Model(inputs=[inputs], outputs=[outputs])

  #NOTE: Compile the model in the main program to make it easy to test with various loss functions
  #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  #model.summary()

  return model


def get_model(n_classes, img_height, img_width, img_channels):
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=img_height, IMG_WIDTH=img_width, IMG_CHANNELS=img_channels)


def load_masks_and_images(image_folder, mask_folder):
    #file types
    file_extensions = ["png"] #["jpg", "JPG", "jpeg", "png"]

    #Get training images and mask paths then sort
    train_images_fp = []
    train_masks_fp = []
    for directory_path in glob.glob(image_folder):
        print(directory_path)
        for ext in file_extensions:
            for img_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                train_images_fp.append(img_path)

    for directory_path in glob.glob(mask_folder):
        print(directory_path)
        for ext in file_extensions:
            for mask_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                train_masks_fp.append(mask_path)

    #sort image and mask fps to ensure we have the same order to index
    train_images_fp.sort()
    train_masks_fp.sort()

    #get actual masks and images
    train_images = []
    train_masks = []

    for img_path in train_images_fp:
        img = cv2.imread(img_path, 0)
        train_images.append(img)

    for mask_path in train_masks_fp:
        mask = cv2.imread(mask_path, 0)
        train_masks.append(mask)

    #Convert list to array for machine learning processing
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    return train_images, train_masks


def create_train_test_split(train_images, train_masks, n_classes):
    train_images = np.expand_dims(train_images, axis=3)
    train_images = normalize(train_images, axis=1)
    train_masks_input = np.expand_dims(train_masks, axis=3)

    #Create a subset of data for quick testing
    #Picking 10% for testing and remaining for training
    X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

    #Further split training data t a smaller subset for quick testing of models
    X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

    return X_train, X_test, y_train_cat, y_test_cat, y_train, y_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", required=True, default = 'multiclass_unet.hdf5', help="Directory containing all folders with original size images.")
    parser.add_argument("--image_path", required=True, help="Directory where images are stored. ex: /User/micheller/data/images_256_256")
    parser.add_argument("--mask_path", required=True, help="Directory where masks are stored. ex: /User/micheller/data/masks_256_256")
    return parser.parse_args()

def main():

    args = parse_args()

    #load our images and masks that we're using to train our unet model
    IMAGE_PATH = args.image_path + "/*" #'/content/drive/MyDrive/annotation_data/images_256_256/*'
    MASK_PATH = args.mask_path + "/*" #'/content/drive/MyDrive/annotation_data/masks_256_256/*'
    train_images, train_masks = load_masks_and_images(IMAGE_PATH, MASK_PATH)

    #split our dataset into train and test sests
    n_classes=11 #Number of classes for segmentation
    X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = create_train_test_split(train_images, train_masks, n_classes)

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    #train model
    model = get_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train_cat,
                        batch_size = 16,
                        verbose=1,
                        epochs=50,
                        validation_data=(X_test, y_test_cat),
                        shuffle=False)

    #save model
    model_save_path = args.model_save_path
    model.save(model_save_path)

    #print accuracy
    _, acc = model.evaluate(X_test, y_test_cat)
    print("Accuracy is = ", (acc * 100.0), "%")

    #print IOU
    y_pred=model.predict(X_test)
    y_pred_argmax=np.argmax(y_pred, axis=3)
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    return


if __name__ == "__main__":
    main()


