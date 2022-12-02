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
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import trt_pose.coco


from loguru import logger


MODEL_WEIGHTS = 'hand_pose_resnet18_att_244_244.pth'
TRT = False
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
        from preprocessdata import preprocessdata
        self.preprocessdata = preprocessdata(topology, num_parts)

        self.model = resnet18_baseline_att(num_parts, 2 * num_links).eval()
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS))
        logger.info(f'load model pose regression from {MODEL_WEIGHTS}')

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
    def __init__(self, model_weights:str) -> None:
        from preprocessdata import preprocessdata
        self.preprocessdata = preprocessdata(topology, num_parts)

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        import pickle
        self.clf = make_pipeline(
            StandardScaler(), SVC(gamma='auto', kernel='rbf'))
        filename = 'svmmodel.sav'
        with open(filename, 'rb') as f:
            self.clf = pickle.load(f)
        logger.info(f'load model SVM classifier from {filename}')

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


def draw_gesture(img: np.array, name_gesture: str) -> np.array:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    org = (50, 50)
    thickness = 2
    fontScale = 0.5
    img = cv2.putText(img, name_gesture, org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return img


def draw_joints(image, joints) -> np.array:
    for i in joints:
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 1)
    cv2.circle(image, (joints[0][0], joints[0][1]), 2, (255, 0, 255), 1)
    for i in hand_pose['skeleton']:
        if joints[i[0]-1][0] == 0 or joints[i[1]-1][0] == 0:
            break
        cv2.line(image, (joints[i[0]-1][0], joints[i[0]-1][1]),
                 (joints[i[1]-1][0], joints[i[1]-1][1]), (0, 255, 0), 1)

    return image

if __name__ == '__main__':
    pass
