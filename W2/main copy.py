import sys
import os
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
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# FASTER RCNN
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")


# SET THRESHOLD FOR SCORING
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model



# FASTER RCNN

predictor = DefaultPredictor(cfg)




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



def load_kitty_instances_txt(path, classes=KITTY_MOTS_CLASSES):
    
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
        cont=0
        for line in f:
            gt = line.strip().split(' ')
            obj_id = gt[2]
            class_id = int(obj_id) // 1000
            instance_id = int(obj_id) % 1000

            # read RLE with pycocotools.mask
            # https://stackoverflow.com/questions/76011794/how-to-decode-a-coco-rle-binary-mask-to-an-image-in-javascript
            bbox = {
                "size": [int(gt[3]), int(gt[4])], # el width y el height
                "counts": gt[5],                  # el string rle
            }

            annotations.append({
                "obj_id": obj_id,
                "class_id": class_id,
                "instance_id": instance_id,
                "bbox": mask.decode(bbox),
            })

            print(gt)
            print(annotations[-1])

            cont+=1
            if cont==2: break
    return annotations




if __name__ == "__main__":

    #save_sequence_inferences(predictor, PATH_TRAINING_SET, SAVE_PATH_TRAIN_INFERENCES_KM)
    
    gt_path = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt"
    load_kitty_instances_txt(gt_path)
    km_img_0 = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/000000.png"
    save_img(km_img_0)
        











