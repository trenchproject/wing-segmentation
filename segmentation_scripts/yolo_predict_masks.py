import os
import glob
import wget
import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from skimage.draw import polygon2mask

from utils import load_dataset_images

def get_yolo_model():
    '''Download trained yolo v8 model from huggingface and load in weights'''

    ## check if we already have the trained yolo checkpoint file
    if os.path.exists("yolov8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt"):
        checkpoint = "yolov8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt"
    else:
        model_url = "https://huggingface.co/imageomics/butterfly_segmentation_yolo_v8/resolve/main/yolov8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0_best.pt"
        checkpoint = wget.download(model_url)

    model = YOLO(checkpoint)
    return model


def get_mask(r):
    #get the original image
    image = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB)

    #create an empty array to add our masks onto
    segmented_img_full = np.zeros(image[:,:,0].shape, dtype='uint8')

    #get the xyn coordinates and ids of the predicted masks
    predicted_class_ids = r.boxes.cls.tolist()
    xyn_masks = r.masks.xyn

    #build mask for the image
    for i, class_id in enumerate(predicted_class_ids):
        #get coordinates of polygon
        coords = xyn_masks[i]
        
        #polygon2mask expects coordinates in y,x order so reorder
        #since we're using the normalized xy coordinates, we also multiply x and y 
        #by the target image's width and height so that our masks scale and fit the target img
        coords_adj = [[y * image.shape[0], x * image.shape[1]] for [x,y] in coords] 

        #build mask from normalized coords
        polygon = np.array(coords_adj)
        mask = polygon2mask(image[:,:,0].shape, polygon).astype("uint8")

        #assign the class id as the pixel value for the segment such that
        #instead of 1s and 0s, it'll be class_id's and 0's
        mask *= int(class_id) 

        #layer the current segment into one collective mask
        segmented_img_full += mask

        for y in range(0, segmented_img_full.shape[0]):
            for x in range(0, segmented_img_full.shape[1]):
                #if there masks overlap at y,x
                if mask[y, x] != 0 and segmented_img_full[y,x] != 0:
                    #replace that value to avoid pixel values not in our id2label mapping
                    segmented_img_full[y,x] = mask[y,x]
    
    print(predicted_class_ids)
    return segmented_img_full
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Directory containing images we want to predict masks for. ex: /User/micheller/data/jiggins_256_256")
    parser.add_argument("--segmentation_csv", required=False, default = 'dataset_segmentation_info.csv', help="Path to the csv created containing \
                        which segmentation classes are present in each image's predicted mask.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load in our images that we need to get masks for
    dataset_folder = args.dataset + '/*'
    dataset_images, image_filepaths = load_dataset_images(dataset_folder)

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
                                            'right_forewing', 'left_forewing', 
                                            'right_hindwing', 'left_hindwing', 
                                            'ruler', 'white_balance', 
                                            'label', 'color_card', 'body'])
    

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
    results = model.predict(image_filepaths, verbose=False)
    
    # Go through results and build masks where each segmented item is encoded with its class ID as pixel values
    i=0
    for r, fp in zip(results, image_filepaths):
        #get the mask with category id's as pixel values
        mask = get_mask(r) 
        
        #create the path to which the mask will be saved, replicating the folder + naming structure of the input dataset
        mask_path = fp.replace(args.dataset, f'{args.dataset}_masks')
        mask_path = mask_path.replace(f".{fp.split('.')[-1]}", "_mask.png") #replace extension and save mask as a png
        
        #create the folder in which the mask will be saved in if it doesn't exist already
        mask_filename = "/" + mask_path.split('/')[-1] #
        mask_folder = mask_path.replace(mask_filename, "")
        os.makedirs(mask_folder, exist_ok=True)
        
        #save mask with cv2 to preserve pixel categories
        print(f"Mask path:{mask_path}")
        cv2.imwrite(mask_path, mask)

        #enter relevant segmentation data for the image in our dataframe
        classes_in_image = np.unique(mask)
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

    # Save csv containing information about segmentation masks per each image
    dataset_segmented.to_csv(args.segmentation_csv, index=False)
    
    return


if __name__ == "__main__":
    main()