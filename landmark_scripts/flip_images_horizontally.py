import os
import glob
import cv2
import matplotlib.pyplot as plt
import argparse
import shutil

def flip_images(source_folder, dest_folder, color_option=1):
    #go through each species subfolder
    file_extensions = ["jpg", "JPG", "jpeg", "png"]
    for ext in file_extensions:
        for img_path in glob.glob(os.path.join(source_folder, f"*.{ext}")):

            #create the new path to save the image under its wing folder
            wing_path = img_path.replace(source_folder, dest_folder)
            print(wing_path)
            
            #load the actual image file
            img = cv2.imread(img_path, color_option)

            #flip the image
            img_h = cv2.flip(img, 1)

            #save the flipped image in the new location
            saved_img = cv2.imwrite(wing_path, img_h)

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to folder containing cropped wing images. Ex. /Users/michelle/cropped_wings")
    return parser.parse_args()

def main():

    args = parse_args()

    ## Note: for our MLmorph models, we need to flip the left fw's and left hw's only
    # left_forewings = '/Users/michelleramirez/Documents/butterflies/Appendometer/jiggins-image-examples/left_forewing'
    # left_forewings_flipped = '/Users/michelleramirez/Documents/butterflies/Appendometer/jiggins-image-examples/left_forewing_flipped'

    #directories where flipped images will be stored
    flipped = args.input_dir + '_flipped'
    os.makedirs(flipped, exist_ok=True)

    #resave images in the cropped_wings_dir to their respective folders based on their titles
    flip_images(args.input_dir, flipped)

    return

if __name__ == "__main__":
    main() 