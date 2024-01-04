from PIL import Image
import numpy as np
import argparse
import cv2
import glob 
import os


def resize_images(source_image_folder, resized_image_folder, file_extensions, image_size=(256,256)):
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
            image = Image.fromarray(image) #convert img back to PIL Image to save
            image.save(f"{resized_image_folder}/{image_name}.png") #save as a png image to avoid pixel val changes
    
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to folder containing images to resize. Ex: /Users/michelleramirez/Documents/butterflies/annotation_data/body_attached")
    parser.add_argument("--output", required=True, help="Path to outut folder where resized images will be stored in")
    parser.add_argument("--resize_dim", required=True, nargs='+', help="(x,y) dimensions to resize all your images into. Input format: --resize_dim 256 256")
    return parser.parse_args()

def main():
    args = parse_args() 
    file_extensions = ['.jpg', '.JPG', '.png', '.JPEG'] #since there are different extensions in our dataset
    
    image_size = (int(args.resize_dim[0]), int(args.resize_dim[1]))
    resize_images(args.source, args.output, file_extensions, image_size)


if __name__ == "__main__":
    main()

