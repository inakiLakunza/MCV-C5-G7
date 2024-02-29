import sys
import os
import json
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *

import numpy as np
import cv2
import tqdm
import glob
import pickle

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import torch

from pathlib import Path



print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f'The device we will be working on is: {device}')


PATH_PARENT_DIRECTORY = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS"
PATH_TRAINING_SET = os.path.join(PATH_PARENT_DIRECTORY, "training", "image_02")
PATH_TEST_SET = os.path.join(PATH_PARENT_DIRECTORY, "testing", "image_02")
PATH_INSTANCES = os.path.join(PATH_PARENT_DIRECTORY, "instances")
PATH_INSTANCES_TXT = os.path.join(PATH_PARENT_DIRECTORY, "instances_txt")

SAVE_PATH_TRAIN_INFERENCES_KM = "/ghome/group07/C5-W2/task_c/mask/train_inferences_KM"


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

# MASK RCNN
model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# FASTER RCNN
#model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"


cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
# SET THRESHOLD FOR SCORING
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model



# FASTER RCNN

predictor = DefaultPredictor(cfg)


KITTY_MOTS_CLASSES = {
    0:'Car',
    1:'Pedestrian'

}

COCO_CLASSES = {}
for i, name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes):
    COCO_CLASSES[i] = name


# TASK a) and b) 
#------------------------------------------------------------------------------------
'''
try_img_path = "/ghome/group07/mcv/datasets/C5/COCO/train2014/COCO_train2014_000000000009.jpg"
try_img = cv2.imread(try_img_path)
output = predictor(try_img)

print(output["instances"].pred_classes)
print(output["instances"].pred_boxes)

save_img(try_img, output, "try_img.png", cfg)
'''
#------------------------------------------------------------------------------------

# https://stackoverflow.com/questions/56150230/how-to-find-top-left-top-right-bottom-left-right-coordinates-in-2d-mask-where


def convert_bbox_format(bbox, width, height):
    # obtain x_left, y_top, x_right, y_bottom = bbox from image (bbox)
    nonzero_indices = np.nonzero(bbox)
    y_min = np.min(nonzero_indices[0])
    x_min = np.min(nonzero_indices[1])
    return int(x_min), int(y_min), width, height

def kitty_gt_to_json(path, image_path, classes=KITTY_MOTS_CLASSES):
    """
    Load a KITTY sequence into COCO sequence
    """
    
    # Create categories in COCO format
    categories = []
    for i, name in enumerate(classes):
        categories.append({
            "id": i+1,
            "name": name,
        })
    
    # Create annotations in COCO format
    annotations = []
    with open(path) as f:
        image_ids = []
        cont=0
        for line in f:
            gt = line.strip().split(' ')
            obj_id = gt[2]
            class_id = int(obj_id) // 1000
            instance_id = int(obj_id) % 1000

            # read RLE with pycocotools.mask
            # https://stackoverflow.com/questions/76011794/how-to-decode-a-coco-rle-binary-mask-to-an-image-in-javascript
            bbox = mask.decode({
                "size": [int(gt[3]), int(gt[4])],
                "counts": gt[5],
            })

            annotations.append({
                "id": gt[1],
                "image_id": gt[0],
                "category_id": obj_id,
                "segmentation": gt[5], # RLE
                "bbox": convert_bbox_format(bbox, int(gt[3]), int(gt[4])),
                "area": int(gt[3]) * int(gt[4]),
                "iscrowd": 0,
            })
            image_ids.append(gt[0])
            cont=+1
    
    # Images in COCO format
    images = []
    cont=0
    filenames = glob.glob(os.path.join(image_path, "*.png"))
    for i, filename in enumerate(sorted(filenames)):
        print(filename)
        height, width = cv2.imread(filename).shape[:2]
        images.append({
            "id": image_ids[i],         
            "file_name": filename,  
            "height": height,     
            "width": width,      
        })
        cont += 1

    # Dump JSON
    with open('./annotations.json', 'w') as json_file:
        json.dump({"images": images, "categories": categories, "annotations": annotations}, json_file)

    return annotations




if __name__ == "__main__":

    #save_sequence_inferences(predictor, PATH_TRAINING_SET, SAVE_PATH_TRAIN_INFERENCES_KM)
    image_path = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/instances/0000/"
    gt_path = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt"
    kitty_gt_to_json(gt_path, image_path)

    path_train = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/"
    path_test = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/testing/image_02/0000/"
    annotations = "/ghome/group07/C5-W2/annotations.json"
    
    register_json("DatasetKittyTrain", annotations, path_train)
    register_json("DatasetKittyTest", annotations, path_test)

    results_path = "./outputs/"
    evaluator = COCOEvaluator("DatasetKittyTrain", cfg, False, output_dir=results_path)


    val_loader = build_detection_test_loader(cfg, "DatasetKittyTest")
    print('---------------------------------------------------------')
    print(model)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    #km_img_0 = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/000000.png"
    #save_img(km_img_0)
        











