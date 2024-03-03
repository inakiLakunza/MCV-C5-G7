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




cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

# MASK RCNN
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#SAVE_PATH_TRAIN_INFERENCES_KM = "/ghome/group07/C5-W2/task_c/mask/train_inferences_KM"
#SAVE_PATH_TEST_INFERENCES_KM = "/ghome/group07/C5-W2/task_c/mask/test_inferences_KM"

# FASTER RCNN
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
SAVE_PATH_TRAIN_INFERENCES_KM = "/ghome/group07/C5-W2/task_c/faster/train_inferences_KM"
SAVE_PATH_TEST_INFERENCES_KM = "/ghome/group07/C5-W2/task_c/faster/test_inferences_KM"


# SET THRESHOLD FOR SCORING
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model




predictor = DefaultPredictor(cfg)


'''
# TASK a) and b) 
#------------------------------------------------------------------------------------

#try_img_path = "/ghome/group07/mcv/datasets/C5/COCO/train2014/COCO_train2014_000000000009.jpg"
try_img_path = '/ghome/group07/mcv/datasets/C5/KITTI-MOTS/training/image_02/0019/000002.png'
try_img = cv2.imread(try_img_path)
output = predictor(try_img)

print(output["instances"].pred_classes)
print(output["instances"].pred_boxes)

save_img(try_img, output, "try3.png", cfg)

#------------------------------------------------------------------------------------
'''





if __name__ == "__main__":
    save_sequence_inferences(predictor, PATH_TRAINING_SET, SAVE_PATH_TRAIN_INFERENCES_KM)
    save_sequence_inferences(predictor, PATH_TEST_SET, SAVE_PATH_TEST_INFERENCES_KM)












