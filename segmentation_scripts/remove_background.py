import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse


def load_dataset_images(dataset_path, color_option=0):
    '''Load in actual images from filepaths from all subfolders in the provided dataset_path'''
    #file types
    file_extensions = ["png"] #["jpg", "JPG", "jpeg", "png"]

    #Get training images and mask paths then sort
    image_filepaths = []
    for directory_path in glob.glob(dataset_path):
        print(directory_path)
        for ext in file_extensions:
            for img_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                image_filepaths.append(img_path)


    #sort image and mask fps to ensure we have the same order to index
    image_filepaths.sort()

    #get actual masks and images
    dataset_images = []

    for img_path in image_filepaths:
        if color_option == 0:
            #read image in grayscale
            img = cv2.imread(img_path, color_option)
        elif color_option == 1:
            #read in color and reverse order to RGB since opencv reads in BGR
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset_images.append(img)

    #Convert list to array for machine learning processing
    dataset_images = np.array(dataset_images)
    return dataset_images, image_filepaths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dataset_path", required=True, help="Directory containing images we have masks for. ex: /User/micheller/data/jiggins_256_256")
    parser.add_argument("--mask_dataset_path", required=True, help="Directory containing masks for images.")
    parser.add_argument("--main_folder_name", required=True, help="JUST the main FOLDER NAME containing all subfolders/images. ex: jiggins_256_256")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # load in our images
    image_dataset_folder = args.image_dataset_path + '/*'
    dataset_images, image_filepaths = load_dataset_images(image_dataset_folder, 1)

    #load in our masks
    mask_dataset_folder = args.mask_dataset_path + '/*'
    dataset_masks, mask_filepaths = load_dataset_images(mask_dataset_folder)
    
    #create a dataframe to store all metadata associated with predicted masks
    classes = {0: 'background',
            1: 'generic',
            2: 'right_forewing',
            3: 'left_forewing',
            4: 'right_hindwing',
            5: 'left_hindwing',
            6: 'ruler',
            7: 'white_balance',
            8: 'label',
            9: 'color_card',
            10: 'body'}
    

    errors = []
    for (image, mask), fp in zip(zip(dataset_images, dataset_masks), image_filepaths): 
        #remove background from color image
        img_removed_background = cv2.bitwise_and(image, image, mask=mask)

        #save the image with the removed background under its own folder
        bck_img_path = fp.replace(args.main_folder_name, f'{args.main_folder_name}_removed_background')
        fn = "/" + bck_img_path.split('/')[-1]
        background_folder = bck_img_path.replace(fn, "")
        os.makedirs(background_folder, exist_ok=True)

        #save the resized cropped wings to their path
        try:
            cv2.imwrite(bck_img_path, cv2.cvtColor(img_removed_background, cv2.COLOR_RGB2BGR))
        except FileNotFoundError:
            errors.append(bck_img_path)
            
    print('The following images could encountered errors during background removal:', errors)
    return


if __name__ == "__main__":
    main()