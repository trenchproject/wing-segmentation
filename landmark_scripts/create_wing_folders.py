import os
import glob
import cv2
import matplotlib.pyplot as plt
import argparse
import shutil

CLASSES = {0: 'background',
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


def get_wing_path(main_folder, img_path):
    #create the new path to save the image under its wing folder
    wing_class = int(img_path.split('_wing_')[-1].split('.png')[0])
    wing_folder = CLASSES[wing_class]
    image_name = img_path.split('_wing_')[0].split('/')[-1]
    wing_path = f"{main_folder}_{wing_folder}/{image_name}.png"
    print('wing path:', wing_path)
    return wing_path

def write_images_to_wing_folders(main_folder):
    #go through each species subfolder
    for directory_path in glob.glob(main_folder + '/*'):
        if os.path.isfile(directory_path):
            img_path = directory_path

            #get new path where the cropped wing will be stored
            wing_path = get_wing_path(main_folder, img_path)

            #copy the img to its new wing folder
            shutil.copy(img_path, wing_path)

        elif os.path.isdir(directory_path):
            #go through each image in the current species subfolder
            for img_path in glob.glob(os.path.join(directory_path, "*.png")):
                
                #get new path where the cropped wing will be stored
                wing_path = get_wing_path(main_folder, img_path)
                
                #copy the img to its new wing folder
                shutil.copy(img_path, wing_path)

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to folder containing cropped wing images. Ex. /Users/michelle/cropped_wings")
    return parser.parse_args()

def main():

    args = parse_args()

    #directory of where all the cropped wings (not seperated by type of wing) are saved
    cropped_wings_dir = args.input_dir
    # '/Users/michelleramirez/Documents/butterflies/ml-morph/jiggins-image-examples/cropped_wings_256_256/jiggins/jiggins_data_256_256/*'

    #create seperate directories for each type of wing, which we will seperate our wing images into
    right_forewings = args.input_dir + '_right_forewing' #'/Users/michelleramirez/Documents/butterflies/ml-morph/jiggins-image-examples/right_forewing'
    right_hindwings = args.input_dir + '_right_hindwing' #'/Users/michelleramirez/Documents/butterflies/ml-morph/jiggins-image-examples/right_hindwing'
    left_forewings  = args.input_dir + '_left_forewing'  #'/Users/michelleramirez/Documents/butterflies/ml-morph/jiggins-image-examples/left_forewing'
    left_hindwings  = args.input_dir + '_left_hindwing'  #'/Users/michelleramirez/Documents/butterflies/ml-morph/jiggins-image-examples/left_hindwing'

    os.makedirs(right_forewings, exist_ok=True)
    os.makedirs(right_hindwings, exist_ok=True)
    os.makedirs(left_forewings, exist_ok=True)
    os.makedirs(left_hindwings, exist_ok=True)

    #resave images in the cropped_wings_dir to their respective folders based on their titles
    write_images_to_wing_folders(cropped_wings_dir)

    return

if __name__ == "__main__":
    main() 