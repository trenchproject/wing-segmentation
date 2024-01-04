from PIL import Image
import numpy as np
import argparse
import cv2
import glob 
import os

def resize_images(dataset_path, resized_image_folder, main_folder_name, file_extensions, image_size=(256, 256)):
    #create a new folder to save resized images to
    os.makedirs(resized_image_folder.replace("*", ""), exist_ok=True)

    print('starting...')
    #begin resizing
    not_resized = []
    for species_folder_path in glob.glob(dataset_path):
        print('FOLDER', species_folder_path)
        for extension in file_extensions:
            dir = species_folder_path + f"/*.{extension}"
            os.makedirs(species_folder_path.replace(main_folder_name, f'{main_folder_name}_{image_size[0]}_{image_size[1]}'), exist_ok=True) #make sure the save dir exists
            for filename in glob.glob(dir): #os.path.join(species_folder_path, f"/*.{extension}")

                try:
                    image = Image.open(filename)
                    image = np.array(image) #convert to numpy to resize
                    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

                    save_filename = filename.replace(main_folder_name, f'{main_folder_name}_{image_size[0]}_{image_size[1]}')
                    save_filename = save_filename.replace(save_filename.split('.')[-1], 'png')
                    print(save_filename)

                    image = Image.fromarray(image) #convert img back to PIL Image to save
                    image.save(save_filename) #save as a png image to avoid pixel val changes
                except OSError:
                    not_resized.append(filename)
    
    return not_resized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to main folder containing all species subfolders of jiggins dataset.")
    parser.add_argument("--output", required=True, help="Main directory to download all subfolders and their resized images into.")
    parser.add_argument("--resize_dim", required=True, nargs='+', help="(x,y) dimensions to resize all your images into")
    parser.add_argument("--main_source_folder_name", required=True, help="Folder name to modify in creating new paths")
    return parser.parse_args()

def main():

    #get arguments from commandline
    args = parse_args() 
    data = args.source + '/*'
    data_resized = args.output + '/*'
    image_size = (int(args.resize_dim[0]), int(args.resize_dim[1]))
    file_extensions = ['jpg', 'JPG', 'png', 'JPEG']
    main_folder_name_to_replace = args.main_source_folder_name

    #resize and save images to input dim (ex. (256,256))
    print(f'image_size={image_size}')
    list_of_failed_resized = resize_images(data, data_resized, main_folder_name_to_replace, file_extensions, image_size)

    print(f"The following files failed to resize: {list_of_failed_resized}")

    return

if __name__ == "__main__":
    main() 