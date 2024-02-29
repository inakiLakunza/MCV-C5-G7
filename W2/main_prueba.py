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
    1:'Pedestrian',
}

COCO_CLASSES = {}
for i, name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes):
    COCO_CLASSES[i] = name





def from_KITTY_to_COCO(path, part):
    
    COCO_classes = {
        0: 81,              # Background anywhere
        1: 2,               # Car to Car
        2: 0,               # Pedestrian to Person
        10: 71
    }
    
    with open('./configs/dataset_split.json') as f_splits:
        sequences = json.load(f_splits)[part]

    if part == "val":
        part = "training"

    sequence_dir = os.path.join(path, part, "image_02")
    
    
    annotations = []
    
    for seq in Path(sequence_dir).glob("*"):
        # Get name of the file directory
        sequence = seq.parts[-1]

        # Ensure the sequence belongs to the selected partition
        if sequence not in sequences:
            continue

        with open(os.path.join(path, "instances_txt", sequence + ".txt")) as f_ann:
            gt = pd.read_table(
                f_ann,
                sep=" ",
                header=0,
                names=["frame", "obj_id", "class_id", "height", "width", "rle"],
                dtype={"frame": int, "obj_id": int, "class_id": int,
                       "height": int, "width": int, "rle": str}
            )
        for img_path in Path(seq).glob("*.png"):
            img_name = img_path.parts[-1]
            frame = int(img_path.parts[-1].split('.')[0])
            frame_gt = (gt[gt["frame"] == frame])

            if len(frame_gt) == 0:
                continue

            ann = []
            for _, obj_id, class_id, height, width, rle in frame_gt.itertuples(index=False):

                # reads rle and decodes it with cocotools
                mask = {
                    "counts": rle.encode('utf8'),
                    "size": [height, width],
                }

                bbox = toBbox(mask).tolist()

                ann.append({
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": COCO_classes[class_id],
                    "segmentation": mask,
                    "keypoints": [],
                    "iscrowd": 0
                })

            annotations.append({
                "file_name": str(img_path),
                "height": frame_gt.iloc[0]["height"],
                "width": frame_gt.iloc[0]["width"],
                "image_id": int(f"{sequence}{frame:05}"),
                "sem_seg_file_name": str(os.path.join(path, "instances", sequence, img_name)),
                "annotations": ann
            })

    return annotations





if __name__ == "__main__":

    #save_sequence_inferences(predictor, PATH_TRAINING_SET, SAVE_PATH_TRAIN_INFERENCES_KM)
    image_path = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/instances/0000/"
    gt_path = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt"

    path_train = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/"
    path_test = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/testing/image_02/0000/"
    annotations = "/ghome/group07/C5-W2/annotations.json"
    
    register_json("DatasetKittyTrain", annotations, path_train)
    register_json("DatasetKittyTest", annotations, path_test)

    coco_names = [""] * 81
    coco_names[80] = "background"
    coco_names[0] = "pedestrian"
    coco_names[2] = "car"    
    coco_names[71] = "sink"


    DATASET_NAME = "KITTI-MOTS-COCO_"
    for d in ['training', 'val']:
        DatasetCatalog.register(DATASET_NAME + d, lambda d=d: from_KITTY_to_COCO(PATH_PARENT_DIRECTORY, d))
        MetadataCatalog.get(DATASET_NAME + d).set(
            thing_classes=coco_names, stuff_classes=coco_names
        )
    metadata = MetadataCatalog.get(DATASET_NAME + "val")

    print('Creating dataset')
    dataset_dicts = from_KITTY_to_COCO(PATH_PARENT_DIRECTORY, 'val')
    image_ids = [x["image_id"] for x in dataset_dicts]

    kitti_meta = MetadataCatalog.get(DATASET_NAME + "val")

    img = cv2.imread(dataset_dicts[0]["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_meta, scale=0.5)
    out = visualizer.draw_dataset_dict(dataset_dicts[0])
    plt.figure(dpi=150)
    plt.axis("off")
    plt.imshow(img)
    plt.show()

    cfg.merge_from_file(model_zoo.get_config_file(model))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.VAL = DATASET_NAME + "val"
    predictor = DefaultPredictor(cfg)


    print('Evaluating model')

    results_model = os.path.join("outputs_prueba", model.split('.')[0].split('/')[0])
    #results_model.mkdir(exist_ok=True)

    evaluator = COCOEvaluator(DATASET_NAME + "val", output_dir=str(results_model))
    val_loader = build_detection_test_loader(cfg, DATASET_NAME + "val")

    print(inference_on_dataset(predictor.model, val_loader, evaluator))







