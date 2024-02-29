import sys
import os
import json
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *

import numpy as np
import tqdm
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path

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
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

from pycocotools.mask import toBbox

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

FT_DATASET_NAME = "KITTI-MOTS_"



if __name__ == "__main__":

    COCO_classes = {
        0: 81,              # Background anywhere
        1: 2,               # Car to Car
        2: 0,               # Pedestrian to Person
        10: 71
    }

    kitti_names = [""] * 11
    kitti_names[0] = "background"
    kitti_names[1] = "car"
    kitti_names[2] = "pedestrian"
    kitti_names[10] = "ignore"


    for d in ['training', 'val']:
        DatasetCatalog.register(FT_DATASET_NAME + d, lambda d=d: get_KITTI(PATH_PARENT_DIRECTORY, d))
        MetadataCatalog.get(FT_DATASET_NAME + d).set(
            thing_classes=kitti_names, stuff_classes=kitti_names
        )

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

    # MASK RCNN
    model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    # FASTER RCNN
    #model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

    cfg.defrost()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    # SET THRESHOLD FOR SCORING
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model



    cfg.DATASETS.TRAIN = (FT_DATASET_NAME + "training",)
    cfg.DATASETS.TEST = (FT_DATASET_NAME + "val",)

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATALOADER.NUM_WORKERS = 4



    output_dir = "/ghome/group07/C5-W2/outputs_task_e"
    cfg.OUTPUT_DIR = output_dir

    with open('./configs/task_e_train_config.json') as train_info:
        info = json.load(train_info)
    cfg.SOLVER.BASE_LR = info["lr"]
    cfg.SOLVER.MAX_ITER = info["max_iter"]
    cfg.SOLVER.BATCH_SIZE_PER_IMAGE = info["batch_size"]
    cfg.SOLVER.IMS_PER_BATCH = info["imgs_per_batch"]
    cfg.SOLVER.CHECKPOINT_PERIOD = info["checkpoint_period"]

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(kitti_names)

    # TRAIN THE MODEL
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



    # EVALUATE THE MODEL
    evaluator = COCOEvaluator(
                FT_DATASET_NAME + "val",
                output_dir=str(output_dir),
    )

    predictor = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, FT_DATASET_NAME + "val")


    print(inference_on_dataset(predictor.model, val_loader, evaluator))








