from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import json


def anns_to_multiclass_mask(coco_json_file, save_mask_dir, mask_size=(128,128), save=False):
    '''Function to convert COCO Annotations to Masks. Masks are saved under the original 
    image name in the provided save_mask_dir. Masks are also resized'''
    
    #load the annotations with COCO
    coco = COCO(coco_json_file)
    image_ids = coco.imgs.keys()
    category_ids = coco.getCatIds()

    #make sure our folder to save masks exists
    os.makedirs(save_mask_dir, exist_ok=True)

    for img_id in image_ids:
        #get the annotations for the img
        img = coco.imgs[img_id]
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=category_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        #get and scale the masks (by category) belonging to each annotation
        multiclass_mask = (anns[0]['category_id']) * coco.annToMask(anns[0])
        multiclass_mask = cv2.resize(multiclass_mask, mask_size, interpolation=cv2.INTER_NEAREST)
        for i in range(1, len(anns)):
            category_mask = (anns[i]['category_id']) * coco.annToMask(anns[i])
            category_mask = cv2.resize(category_mask, mask_size, interpolation=cv2.INTER_NEAREST)
            multiclass_mask += category_mask

            for y in range(0, category_mask.shape[0]):
                for x in range(0, category_mask.shape[1]):
                    #if there masks overlap at y,x
                    if category_mask[y, x] != 0 and multiclass_mask[y,x] != 0:
                        #replace that value to avoid pixel values not in our id2label mapping
                        multiclass_mask[y,x] = category_mask[y,x]

        #save the mask
        if save:
            name = img['file_name'].split('.')[0]
            fp = f"{save_mask_dir}/{name}.png" #save as png to avoid changing pixel values
            im = Image.fromarray(multiclass_mask)
            im.save(fp)


def main():
    # Get (RESIZED) Masks for All COCO JSON Files
    mask_dir = '/Users/michelleramirez/Documents/butterflies/annotation_data/masks_128_128'

    body_attached_coco_json = '/Users/michelleramirez/Documents/butterflies/annotation_data/coco_annotations_cvat/body_attached_coco/instances_default.json'
    damaged_coco_json = '/Users/michelleramirez/Documents/butterflies/annotation_data/coco_annotations_cvat/damaged_coco/instances_default.json'
    dorsal_coco_json = '/Users/michelleramirez/Documents/butterflies/annotation_data/coco_annotations_cvat/dorsal_coco/instances_default.json'
    incomplete_coco_json = '/Users/michelleramirez/Documents/butterflies/annotation_data/coco_annotations_cvat/incomplete_coco/instances_default.json'

    anns_to_multiclass_mask(body_attached_coco_json, f"{mask_dir}/body_attached", (128,128), True)
    anns_to_multiclass_mask(damaged_coco_json, f"{mask_dir}/damaged", (128,128), True)
    anns_to_multiclass_mask(dorsal_coco_json, f"{mask_dir}/dorsal", (128,128), True)
    anns_to_multiclass_mask(incomplete_coco_json, f"{mask_dir}/incomplete", (128,128), True)


if __name__ == "__main__":
    main()