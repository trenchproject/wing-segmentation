import os
import glob
import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO


def load_dataset_images(dataset_path, color_option=0):
    '''Load in actual images from filepaths from all subfolders in the provided dataset_path'''
    #file types
    file_extensions = ["jpg", "JPG", "jpeg", "png"]

    #Get training images and mask paths then sort
    image_filepaths = []
    for directory_path in glob.glob(dataset_path):
        if os.path.isfile(directory_path):
            image_filepaths.append(directory_path)
        elif os.path.isdir(directory_path):
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


def get_yolo_model():
    '''Download trained yolo v8 model from huggingface and load in weights'''


    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", required=True, default = 'multiclass_unet.hdf5', help="Directory containing all folders with original size images.")
    parser.add_argument("--dataset_path", required=True, help="Directory containing images we want to predict masks for. ex: /User/micheller/data/jiggins_256_256")
    parser.add_argument("--main_folder_name", required=True, help="JUST the main FOLDER NAME containing all subfolders/images. ex: jiggins_256_256")
    parser.add_argument("--segmentation_csv", required=True, default = 'segmentation_info.csv', help="Path to the csv created containing \
                        which segmentation classes are present in each image's predicted mask.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load in our images that we need to get masks for
    dataset_folder = args.dataset_path + '/*' #'/content/drive/MyDrive/annotation_data/jiggins/jiggins_data_256_256/*'
    dataset_images, image_filepaths = load_dataset_images(dataset_folder)

    # Main folder name is used to create a new directory under a modified version of the original folder name
    folder_name = args.main_folder_name

    # Get Model
    model = get_yolo_model()

    # Create a dataframe to store all metadata associated with predicted masks
    classes = {0: 'background',
            1: 'right_forewing',
            2: 'left_forewing',
            3: 'right_hindwing',
            4: 'left_hindwing',
            5: 'ruler',
            6: 'white_balance',
            7: 'label',
            8: 'color_card',
            9: 'body'}
    
    dataset_segmented = pd.DataFrame(columns = ['image', 'background', 
                                            'generic', 'right_forewing', 
                                            'left_forewing', 'right_hindwing', 
                                            'left_hindwing', 'ruler', 'white_balance', 
                                            'label', 'color_card', 'body', 'damaged'])
    

    # Leverage GPU if available
    use_cuda = torch.cuda.is_available()
    DEVICE   = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ", DEVICE)

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    # Predict masks on all our images

    # Save csv containing information about segmentation masks per each image
    dataset_segmented.to_csv(args.segmentation_csv, index=False)
    
    return


if __name__ == "__main__":
    main()