# wing-segmentation

This repository contains the scripts necessary to extract the following components using a UNET segmetnation model from butterfly images:
- wings (right forewing, right hindwing, left forewing, left hindwing) 
- ruler
- label
- color palette


## 1. Preprocessing

Before predicting segmentation masks for your images, the images will need to be resized to 256x256. 

To resize your images, you can use either the `resize_images_flat_dir.py` or the `resize_images_subfolders.py` file in the `preprocessing_scripts` folder. The only difference between the two is the assumed folder structures. Both scripts will create a new directory containing your resized images such that the original images are not overwritten/modified. 

### Option 1: Resizing with `resize_images_flat_dir.py`
command: 
```
python3 resize_images_flat_dir.py --source /path/to/source/image/dataset/folder --output /path/to/new/folder/to/store/images --resize_dim 256 256
```

The script `resize_images_flat_dir.py` expects your source folder structure to look as such:
```
|-- Source_Image_Folder
|   |-- image_1.jpg
|   |-- image_2.jpg
|   |-- image_3.jpg
|   |-- image_4.jpg
|   |-- image_5.jpg
```

### Option 2: Resizing with `resize_images_subfolders.py`
command: 
```
python3 resize_images_subfolders.py --source /path/to/source/image/dataset/folder --output /path/to/new/folder/to/store/images --resize_dim 256 256
```

The script `resize_images_subfolders.py` expects your source folder structure to look as such:

```
|-- Source_Image_Folder
|   |--species_folder_1
|   |   |-- image_1.jpg
|   |   |-- image_2.jpg
|   |--species_folder_2
|   |   |-- image_1.jpg
|   |   |-- image_2.jpg
|   |--species_folder_3
|   |   |-- image_1.jpg
|   |   |-- image_2.jpg

```

## 2. Predicting Segmentation Masks

After resizing your images to 256 x 256 (required dimensions for trained unet model), you will run the `unet_predict_masks.py` script in the `segmentation_scipts` folder. The result will be a new folder containing all the segmentation masks for each of your images in the input directory.

command: 

```
python3 unet_predict_masks.py --model_save_path /path/to/unet_butterflies_256_256.hdf5 --dataset_path /path/to/your/256x256/resized/images --main_folder_name name_of_source_image_folder --segmentation_csv /path/where/to/store/segmentation_info.csv

```

Arguments explained: 

`--model_save_path` is the location of where you downloaded and stored the pretrained unet model (you can retrieve the model here: [unet_butterflies_256_256.hdf5](https://huggingface.co/imageomics/butterfly_segmentation_unet/blob/main/unet_butterflies_256_256.hdf5))

`--dataset_path` is the full path of where your folder containing your resized images obtained in **step 1**. (example: /User/micheller/data/jiggins_256_256)

`--main_folder_name` is ONLY the name of the source folder containing your resized images, not the full path. (exmaple: jiggins_256_256) 

`segmentation_csv` is the path location at which you want to store the csv that gets created detailing which segmentation categories exist in the mask generated for each image. (Optional. Default segmentation.csv will be saved in the same directory as this script.)

## 3. Using Segmentation Masks to Extract Wings from Images

After obtaining masks for our images, we can crop out the forewings and hindwings by running the following `crop_wings_out.py` script in the `segmentation_scripts` folder:

```
python3 crop_wings_out.py --image_dataset_path /path/to/image/dataset --mask_dataset_path /path/to/segmentation/masks --output_folder /path/to/folder/where/we/store/cropped/wing/results
```