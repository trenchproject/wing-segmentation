from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse

from train_unet import get_model


def load_dataset_images(dataset_path):
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
        img = cv2.imread(img_path, 0)
        dataset_images.append(img)

    #Convert list to array for machine learning processing
    dataset_images = np.array(dataset_images)
    return dataset_images, image_filepaths


def save_segmented_wings(test_img, predicted_img, cropped_dim=(256, 256)):
  '''Goes through the predicted mask of an image and crops down the original image to each of the 4 available wings
  based on the predicted segmented wing mask'''

  cropped_wings = dict()
  cropped_wings_resized = dict()

  #only search for masks belonging to right/left hindwings and forewings
  for wing_class in [2,3,4,5]:
    img = test_img[:,:, 0]
    mask = np.asarray(predicted_img==wing_class, dtype=int) #0s and 1s

    y_coords = []
    x_coords = []
    for y in range(0, mask.shape[0]):
      for x in range(0, mask.shape[1]):
        if mask[y,x]==1:
          x_coords.append(x)
          y_coords.append(y)

    #our mask is empty - therefore no existing mask for that wing
    if len(x_coords) == 0 and len(y_coords) == 0:
      continue

    #sort the x and y coordinates of our segmented wing mask to crop accordingly
    x_coords.sort()
    y_coords.sort()

    miny = y_coords[0]
    maxy = y_coords[-1]
    minx = x_coords[0]
    maxx= x_coords[-1]

    #get boundaries of segmented mask with some extra room
    if y_coords[0] > 10 and x_coords[0] > 10:
      miny -= 10
      maxy += 10
      minx -= 10
      maxx += 10

    #crop image down to segmented wings
    cropped_result = img[miny:maxy, minx:maxx]
    cropped_result_resized = cv2.resize(cropped_result, cropped_dim, interpolation=cv2.INTER_CUBIC) #resize to provided dimensions

    #store results in dictionary {wing_class: cropped_image}
    cropped_wings[wing_class] = cropped_result 
    cropped_wings_resized[wing_class] = cropped_result_resized #resize to provided dimensions

  #return both original sized and resized cropped wings *just in case*
  return cropped_wings, cropped_wings_resized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", required=True, default = 'multiclass_unet.hdf5', help="Directory containing all folders with original size images.")
    parser.add_argument("--dataset_path", required=True, help="Directory containing images we want to predict masks for. ex: /User/micheller/data/jiggins_256_256")
    return parser.parse_args()


def main():
    args = parse_args()

    # load in our images that we need to get masks for
    dataset_folder = '/content/drive/MyDrive/annotation_data/jiggins/jiggins_data_256_256/*' #args.dataset_path + '/*'
    dataset_images, image_filepaths = load_dataset_images(dataset_folder)

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

        #save the entire predicted mask
        mask_path = fp.replace('jiggins', 'jiggins_masks')
        mask_path = mask_path.replace('.png', '_mask.png')
        mask_fn = "/" + mask_path.split('/')[-1]
        mask_folder = mask_path.replace(mask_fn, "")
        os.makedirs(mask_folder, exist_ok=True)
        plt.imsave(mask_path, predicted_img)

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

        #crop + extract any existing wings
        cropped_wings, cropped_wings_resized = save_segmented_wings(test_img, predicted_img)

        #save each individual wing with a traceable name to the source image 
        #(ex. erato_0001_wing_2.png denotes the image contains the right forewing for erato_0001.png)
        for wing_idx in cropped_wings.keys():
            cropped_wing = cropped_wings[wing_idx]
            cropped_wing_resized = cropped_wings_resized[wing_idx]

            #create path to save the non-resized wing crops to
            cropped_wing_path = fp.replace('/jiggins/', '/cropped_wings/jiggins/').replace('.png', f'_wing_{wing_idx}.png')
            n = "/" + cropped_wing_path.split('/')[-1]
            cropped_wing_folder = cropped_wing_path.replace(n, "")
            os.makedirs(cropped_wing_folder, exist_ok=True)
            
            #create path to save the resized wing crops to
            resized_cropped_wing_path = fp.replace('/jiggins/', '/cropped_wings_256_256/jiggins/').replace('.png', f'_wing_{wing_idx}.png')
            r = "/" + resized_cropped_wing_path.split('/')[-1]
            resized_cropped_wing_folder = resized_cropped_wing_path.replace(r, "")
            os.makedirs(resized_cropped_wing_folder, exist_ok=True)

            #save the non-resized and the resized cropped wings to their respective dirs
            try:
                plt.imsave(cropped_wing_path, cropped_wing)
                plt.imsave(resized_cropped_wing_path, cropped_wing_resized)
            except FileNotFoundError:
                errors.append(cropped_wing_path)

            i += 1

    return


if __name__ == "__main__":
    main()