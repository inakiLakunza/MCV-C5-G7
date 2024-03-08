import sys
import os

import numpy as np
import cv2
import tqdm
import glob
import json
import pickle


import pandas as pd
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from pycocotools.mask import toBbox

import torch
from pathlib import Path


def save_img(img, output, save_path, cfg):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    # [:, :, ::-1] converts from RBG to BGR and vice versa
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

def transformstojson(transformslist):
    # Convert each transform to a dictionary
    transformed_list = []
    for transform in transformslist:
        transform_dict = {transform.__class__.__name__: transform.__dict__}
        transformed_list.append(transform_dict)

    return transformed_list


def save_sequence_inferences(predictor, sequence_path, save_path):
    sequences = glob.glob(sequence_path+"/*")
    sequences.sort()
    res = {}

    for sequence in sequences:
        seq = sequence.split("/")[-1]
        file_names = glob.glob(os.path.join(sequence_path, seq, "*.png"))
        file_names.sort()

        inference_classes = {}
        inference_boxes = {}
        for name in file_names:
            number = name.split("/")[-1][:-4]
            print(f"Predicting img: {number}  in sequence {seq}")
            img = cv2.imread(name)
            output = predictor(img)

            inference_classes[name] = output["instances"].pred_classes
            inference_boxes[name] = output["instances"].pred_boxes

        seq_path = os.path.join(save_path, seq)
        Path(seq_path).mkdir(parents=True, exist_ok=True)

        pkl_save_path_classes = os.path.join(seq_path, "classes.pkl")
        with open(pkl_save_path_classes, 'wb') as file:
            pickle.dump(inference_classes, file)

        pkl_save_path_boxes = os.path.join(seq_path, "boxes.pkl")
        with open(pkl_save_path_boxes, 'wb') as file:
            pickle.dump(inference_boxes, file)


def register_json(name, json_train, path_train):
    register_coco_instances(name, {}, json_train, path_train)


def from_KITTY_to_COCO_taskd(path, part):
    
    COCO_classes = {
        #0: 80,              # Background anywhere
        1: 2,               # Car to Car
        2: 0,               # Pedestrian to Person
        #10: 80
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

                if class_id != 10:

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

def from_KITTY_to_COCO(path, part):
    
    COCO_classes = {
        #0: 80,              # Background anywhere
        1: 2,               # Car to Car
        2: 0,               # Pedestrian to Person
        #10: 80
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

                if class_id != 10:

                    # reads rle and decodes it with cocotools
                    mask = {
                        "counts": rle.decode('utf8'),
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





def get_KITTI(path, part):
    
    
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
                    "category_id": class_id,
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
