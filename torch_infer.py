#!/usr/bin/python3

import cv2
import numpy as np
import json
import time
import os
import torch
from torchvision import transforms
from PIL import Image

from model import resnet18_baseline_att
from trt_pose.parse_objects import ParseObjects
import trt_pose.coco


DEVICE = torch.device('cuda:0')
IMAGE_SIZE = (224, 224)
with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])
with open('preprocess/gesture.json', 'r') as f:
    gesture = json.load(f)
gesture_type = gesture["classes"]


class TorchPose():
    def __init__(self, model_weights: str) -> None:
        # self.parse_objects = ParseObjects(topology, cmap_threshold=0.15, link_threshold=0.15)
        self.parse_objects = ParseObjects(
            topology, cmap_threshold=0.15, link_threshold=0.15)
        from utils import preprocessdata
        self.preprocessdata = preprocessdata(topology, num_parts)

        self.model = resnet18_baseline_att(num_parts, 2 * num_links).eval()
        self.model.load_state_dict(torch.load(model_weights))

        self.model.to(DEVICE)
        self.img_shape = IMAGE_SIZE
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        self.transforms = transforms.Compose(
            transforms=[transforms.ToTensor(), transforms.Normalize(mean, std)])

    @staticmethod
    def prepro(img: np.array) -> np.array:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        return img

    def preprocess(self, img: np.array) -> torch.Tensor:
        img = self.prepro(img)

        img = Image.fromarray(img)
        img = self.transforms(img)
        img = img.unsqueeze(0)
        return img

    def postprocess(self, output: tuple) -> list:
        cmap, paf = output[0].detach().cpu(), output[1].detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        joints = self.preprocessdata.joints_inference(
            self.img_shape, counts, objects, peaks)
        return joints

    def infer(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(DEVICE)
        with torch.no_grad():
            output = self.model(img)
            return output

    def __call__(self, img: np.array) -> list:
        img = self.preprocess(img)
        output = self.infer(img)
        output = self.postprocess(output)
        return output


class SVMClassifier():
    def __init__(self, model_weights: str) -> None:
        from utils import preprocessdata
        self.preprocessdata = preprocessdata(topology, num_parts)

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        import pickle
        self.clf = make_pipeline(
            StandardScaler(), SVC(gamma='auto', kernel='rbf'))
        with open(model_weights, 'rb') as f:
            self.clf = pickle.load(f)

        self.num_frames = 5
        self.res_queue = [self.num_frames]*self.num_frames
        self.text = 'no hand'

    def __call__(self, joints, single_res: bool = False) -> str:
        dist_bn_joints = self.preprocessdata.find_distance(joints)
        gesture = self.clf.predict([dist_bn_joints, [0]*num_parts*num_parts])
        id_gesture = gesture[0]
        if single_res:
            return gesture_type[id_gesture-1]

        self.res_queue.append(id_gesture)
        self.res_queue.pop(0)

        if self.res_queue == [1] * self.num_frames:
            self.text = gesture_type[0]
        elif self.res_queue == [2] * self.num_frames:
            self.text = gesture_type[1]
        elif self.res_queue == [3] * self.num_frames:
            self.text = gesture_type[2]
        elif self.res_queue == [4] * self.num_frames:
            self.text = gesture_type[3]
        elif self.res_queue == [5] * self.num_frames:
            self.text = gesture_type[4]
        elif self.res_queue == [6] * self.num_frames:
            self.text = gesture_type[5]
        elif self.res_queue == [7]*self.num_frames:
            self.text = gesture_type[6]
        return self.text
