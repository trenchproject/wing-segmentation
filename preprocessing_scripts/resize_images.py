from PIL import Image
import numpy as np
import argparse
import cv2
import glob 
import os

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--source_image_dir", required=True, help="Directory containing all folders with original size images.", nargs="+")
#     parser.add_argument("--resized_image_dir", required=True, help="Directory to put new resized images.", nargs="+")
#     return parser.parse_args()

def resize_images(source_image_folder, resized_image_folder, file_extensions=['.jpg'], image_size=(128,128)):
    #create a new folder to save resized images to
    os.makedirs(resized_image_folder, exist_ok=True)

    #begin resizing
    for extension in file_extensions:
        for filename in glob.glob(f'{source_image_folder}/*{extension}'):
            print(filename)
            image = Image.open(filename)
            image = np.array(image) #convert to numpy to resize
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

            image_name = filename.split("/")[-1] #get rid of everything but the name of the file
            image_name = image_name.split(".")[0] #get rid of the extension
            print(f"{resized_image_folder}/{image_name}")
            image = Image.fromarray(image) #convert img back to PIL Image to save
            image.save(f"{resized_image_folder}/{image_name}.png") #save as a png image to avoid pixel val changes
    
    return


def main():
    body_attached_source = '/Users/michelleramirez/Documents/butterflies/annotation_data/body_attached'
    damaged_source = '/Users/michelleramirez/Documents/butterflies/annotation_data/damaged_wings'
    dorsal_source = '/Users/michelleramirez/Documents/butterflies/annotation_data/other_dorsal'
    incomplete_source = '/Users/michelleramirez/Documents/butterflies/annotation_data/incomplete_wing_set'

    body_attached_resized = '/Users/michelleramirez/Documents/butterflies/annotation_data/images_128_128/body_attached'
    damaged_resized = '/Users/michelleramirez/Documents/butterflies/annotation_data/images_128_128/damaged'
    dorsal_resized = '/Users/michelleramirez/Documents/butterflies/annotation_data/images_128_128/dorsal'
    incomplete_resized = '/Users/michelleramirez/Documents/butterflies/annotation_data/images_128_128/incomplete'

    file_extensions = ['.jpg', '.JPG', '.png', '.JPEG'] #since there are different extensions in our dataset
    resize_images(body_attached_source, body_attached_resized, file_extensions)
    resize_images(damaged_source, damaged_resized, file_extensions)
    resize_images(dorsal_source, dorsal_resized, file_extensions)
    resize_images(incomplete_source, incomplete_resized, file_extensions)


if __name__ == "__main__":
    main()