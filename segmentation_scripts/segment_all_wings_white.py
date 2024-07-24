import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse

from utils import read_image_paths, load_batch_images

def batch_indices_np(total, batch_size):
    indices = np.arange(total)
    return np.array_split(indices, np.ceil(total / batch_size))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dataset_path", required=True, help="Directory containing images we have masks for. ex: /User/micheller/data/jiggins_256_256")
    parser.add_argument("--mask_dataset_path", required=True, help="Directory containing masks for images.")
    parser.add_argument("--main_folder_name", required=True, help="JUST the main FOLDER NAME containing all subfolders/images. ex: jiggins_256_256")
    
    return parser.parse_args()

def remove_background_color_based(image, mask):
    # Create an image with white background
    white_background = np.full(image.shape, 255, dtype=np.uint8)

    print('Image type: ', type(image[0,0]), '|| Mask type: ', type(mask[0,0]))
    print(image.shape, mask.shape)

    # Apply the mask to the white background, where mask is not zero, we will fill it with the image's color
    masked_foreground = cv2.bitwise_and(image, image, mask=mask) #mask=255-mask #wings
    background = cv2.bitwise_and(white_background, white_background, mask=mask)
    background = 255 - background

    # Combine the masked foreground with the background
    final_image = cv2.add(masked_foreground, background)

    return final_image

def main():
    args = parse_args()

    # load in our images
    print('Reading in images...')
    image_dataset_folder = args.image_dataset_path + '/*'
    # dataset_images, image_filepaths = load_dataset_images(image_dataset_folder, 1)
    image_filepaths = read_image_paths(image_dataset_folder)[0:5]


    #load in our masks
    print('Reading in masks...')
    mask_dataset_folder = args.mask_dataset_path + '/*'
    # dataset_masks, mask_filepaths = load_dataset_images(mask_dataset_folder)
    mask_filepaths = read_image_paths(mask_dataset_folder)[0:5]

    #create batches of 32
    batches = batch_indices_np(len(image_filepaths), 32)
    
    print('Removing everything but forewings and hindwings...')
    errors = []
    for batch in batches:
        print('Batch: ', batch)
        img_batch_fps = [image_filepaths[i] for i in batch]
        mask_batch_fps = [mask_filepaths[i] for i in batch]

        dataset_images = load_batch_images(img_batch_fps, 1)
        dataset_masks = load_batch_images(mask_batch_fps)

        for (image, mask), fp in zip(zip(dataset_images, dataset_masks), img_batch_fps): 
            # subselect the wing categories in our mask
            print(fp)
            wing_mask = 4* (mask == 4) | 3 * (mask == 3) | 2 * (mask == 2) | 1 *(mask == 1)
            wing_mask  = wing_mask.astype(np.uint8)

            #remove background from color image and replace it with a WHITE background
            img_removed_background = remove_background_color_based(image, wing_mask) #cv2.bitwise_and(image, image, mask=mask)

            #save the image with the removed background under its own folder
            bck_img_path = fp.replace(args.main_folder_name, f'{args.main_folder_name}_wings_only_white_background') #renaming default path temporarily
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