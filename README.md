# wing-segmentation

This repository contains the scripts necessary to extract the following components using a YOLO v8 segmentation model from butterfly images:
- wings (right forewing, right hindwing, left forewing, left hindwing) 
- ruler
- metadata label
- color palette


## 1. Preprocessing (OPTIONAL. Skip to Step 2 if you don't wish to resize your images)

The YOLO model does not require that you resize your images. However, if you wish to resize your images regardless, you can follow the steps below to do so. 

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

## 2. Predicting Segmentation Masks with YOLO

To obtain masks for your set of images, run the `yolo_predict_masks.py` script in the `segmentation_scripts` folder. The result will be a new folder containing all the segmentation masks for each of your images in the input directory.

Command: 

```
python3 wing-segmentation/segmentation_scripts/yolo_predict_masks.py --dataset /path/to/your/images --segmentation_csv /path/where/to/store/segmentation_info.csv

```

Arguments explained: 

`--dataset` is the full path to your folder containing your images you wish to obtain masks for. (Example: /User/micheller/data/jiggins_256_256)

`--segmentation_csv` is the path location at which you want to store the csv that gets created detailing which segmentation categories exist in the mask generated for each image. (Optional. Default segmentation.csv will be saved in the same directory from where you run this script.)

## 3. Using Segmentation Masks to Extract Wings from Images

After obtaining masks for our images, we can crop out the forewings and hindwings by running the following `crop_wings_out.py` script in the `segmentation_scripts` folder:

Command:

```
python3 wing-segmentation/segmentation_scripts/crop_wings_out.py --images /path/to/butterfly/images --masks /path/to/segmentation/masks --output_folder /path/to/save/cropped/wings/to
```

The cropped wing images will be named in this structure: `<original name>_wing_#.png`

The number following `wing` can be mapped as follows:

`1`: right forewing

`2`: left forewing

`3`: right hindwing

`4`: left hindwing

## 4. Creating seperate wing folders and flipping wings

The `landmark_scripts` folder contains python scripts to sort cropped wings into wing folders and flip images horizontally if needed.

Commands:

**Create wing folders**
```
python3 wing-segmentation/landmark_scripts/create_wing_folders.py --input_dir /path/to/folder/where/we/store/cropped/wing/results

```

**Flip images**
```
python3 wing-segmentation/landmark_scripts/flip_images_horizontally.py --input_dir /path/to/wing/category/folder

```