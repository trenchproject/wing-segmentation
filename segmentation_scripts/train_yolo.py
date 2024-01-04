import os
import glob
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.draw import polygon2mask

def main():
    #define path to yaml file containing info about our dataset
    YAML     = '/fs/ess/PAS2136/Butterfly/annotated_cvat_segmentation_data/dataset_splits_yolo/yolo_wing_segmentation.yaml'
    EPOCHS   = 50
    IMG_SIZE = 256

    use_cuda = torch.cuda.is_available()
    DEVICE   = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ", DEVICE)

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)


    # Load a pretrained model
    model = YOLO('yolov8m-seg.pt')

    # Train the model
    save_name = 'yolov8m_shear_10.0_scale_0.5_translate_0.1_fliplr_0.0' ##USE THIS ONE

    results = model.train(data=YAML, 
                        imgsz=IMG_SIZE,
                        epochs=EPOCHS, 
                        batch=16,
                        device=DEVICE,
                        optimizer='auto',
                        verbose=True,
                        val=True,
                        project='/fs/ess/PAS2136/Butterfly/butterfly_image_segmentation/yolo-wing-segmentation/yolo_models',
                        name = save_name,
                        shear=10.0,
                        scale=0.5, 
                        translate=0.1,
                        fliplr = 0.0 # don't add the default flip left-right aug. to our wing images
                        )
    
    return


if __name__ == "__main__":
    main()