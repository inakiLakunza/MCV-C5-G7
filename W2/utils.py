import sys
import os

import numpy as np
import cv2
import tqdm
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import torch
from pathlib import Path


def save_img(img, output, save_path, cfg):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    # [:, :, ::-1] converts from RBG to BGR and vice versa
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])


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
