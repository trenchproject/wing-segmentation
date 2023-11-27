from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse
from PIL import Image

from train_unet import get_model

def load_dataset_images(dataset_path):
    '''Load in actual images from filepaths from all subfolders in the provided dataset_path'''
    #file types
    file_extensions = ["png"] #["jpg", "JPG", "jpeg", "png"]

    #Get training image paths then sort
    image_filepaths = []
    for directory_path in glob.glob(dataset_path):
        print(directory_path)
        if os.path.isfile(directory_path):
            image_filepaths.append(directory_path)
        elif os.path.isdir(directory_path):
            for ext in file_extensions:
                for img_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                    image_filepaths.append(img_path)

    print('IMAGE FILEPATHS:', image_filepaths)

    #sort image and mask fps to ensure we have the same order to index
    image_filepaths.sort()

    #get actual masks and images
    dataset_images = []

    for img_path in image_filepaths:
        img = cv2.imread(img_path, 0)
        dataset_images.append(img)

    #Convert list to array for machine learning processing
    dataset_images = np.array(dataset_images)
    return dataset_images, image_filepaths


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

    # load in our images that we need to get masks for
    dataset_folder = args.dataset_path + '/*' #'/content/drive/MyDrive/annotation_data/jiggins/jiggins_data_256_256/*'
    dataset_images, image_filepaths = load_dataset_images(dataset_folder)

    #main folder name is used to create a new directory under a modified version of the original folder name
    folder_name = args.main_folder_name

    # Load in trained model
    model = get_model(n_classes=11, img_height=256, img_width=256, img_channels=1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(args.model_save_path)

    #preprocess images
    normalized_dataset_images = np.expand_dims(dataset_images, axis=3)
    normalized_dataset_images = normalize(normalized_dataset_images, axis=1)
    
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
    
    dataset_segmented = pd.DataFrame(columns = ['image', 'background', 
                                            'generic', 'right_forewing', 
                                            'left_forewing', 'right_hindwing', 
                                            'left_hindwing', 'ruler', 'white_balance', 
                                            'label', 'color_card', 'body', 'damaged'])
    

    i = 0 #dataframe indexer
    errors = []
    for test_img, fp in zip(normalized_dataset_images, image_filepaths):
        #use the unet model to predict the mask on the image
        test_img_norm = test_img[:,:,0][:,:,None]
        test_image_input = np.expand_dims(test_img_norm, 0)
        prediction = (model.predict(test_image_input))
        predicted_img = np.argmax(prediction, axis=3)[0,:,:]
        print('MASK VALUES:', np.unique(predicted_img))

        #save the entire predicted mask under its own folder
        mask_path = fp.replace(folder_name, f'{folder_name}_masks')
        mask_path = mask_path.replace('.png', '_mask.png')
        mask_fn = "/" + mask_path.split('/')[-1]
        mask_folder = mask_path.replace(mask_fn, "")
        os.makedirs(mask_folder, exist_ok=True)
        
        #save mask with cv2 to preserve pixel categories
        cv2.imwrite(mask_path, predicted_img)

        #enter relevant segmentation data for the image in our dataframe
        classes_in_image = np.unique(predicted_img)
        classes_not_in_image = set(classes.keys()) ^ set(classes_in_image)
        dataset_segmented.loc[i, 'image'] = fp
        
        #enter `1` for all segmentation classes that appear in our mask
        for val in classes_in_image:
            pred_class = classes[val]
            dataset_segmented.loc[i, pred_class] = 1 #class exists in segmentation mask

        #enter `0` for all segmentation classes that were not predicted
        for not_pred_val in classes_not_in_image:
            not_pred_class = classes[not_pred_val]
            dataset_segmented.loc[i, not_pred_class] = 0 #class does not exist in segmentation mask

        i += 1

    #save csv containing information about segmentation masks per each image
    dataset_segmented.to_csv(args.segmentation_csv, index=False)
    
    return


if __name__ == "__main__":
    main()