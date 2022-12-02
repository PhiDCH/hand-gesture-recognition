#!/usr/bin/python3

from torch_infer import TorchPose, SVMClassifier, draw_joints, draw_gesture
import cv2
from PIL import Image
import numpy as np
import unittest
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from loguru import logger

import warnings
warnings.filterwarnings("ignore")


CFG = None


@hydra.main(version_base='1.2', config_path='config', config_name='default_test')
def get_cfg(cfg: DictConfig) -> None:
    global CFG
    CFG = cfg


class MyCase(unittest.TestCase):
    def model(self,):
        self.assertIsInstance(test_load_model_pose(), TorchPose)
        self.assertIsInstance(test_load_model_classify(), SVMClassifier)

    def image(self,):
        joints, img = test_image_pose_estimate()
        self.assertEqual(len(joints), 21)
        name_gesture, img = test_image_pose_classify()
        self.assertEqual(name_gesture, 'stop')
        if CFG.viz:
            Image.fromarray(img).show()

    def video(self,):
        test_video_pose_classify()


def test_load_model_pose():
    model_pose = TorchPose(CFG.model_pose)
    return model_pose


def test_load_model_classify():
    model_classify = SVMClassifier(CFG.model_classify)
    return model_classify


def test_image_pose_estimate() -> list:
    model_pose = TorchPose(CFG.model_pose)
    img = cv2.imread(CFG.img_path)
    joints = model_pose(img.copy())
    if CFG.viz:
        img = draw_joints(model_pose.prepro(img), joints)
    return joints, img


def test_image_pose_classify() -> list:
    model_classify = SVMClassifier(CFG.model_classify)
    joints, img = test_image_pose_estimate()
    name_gesture = model_classify(joints, single_res=True)
    if CFG.viz:
        img = draw_gesture(img, name_gesture)
    return name_gesture, img


def test_video_pose_classify() -> None:
    model_pose = TorchPose(CFG.model_pose)
    model_classify = SVMClassifier(CFG.model_classify)
    if isinstance(CFG.video, int):
        logger.info('test with camera')
    
    cap = cv2.VideoCapture(CFG.video)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            joints = model_pose(frame)
            name_gesture = model_classify(joints)
            if CFG.viz:
                img = draw_joints(model_pose.prepro(frame), joints)
                img = draw_gesture(img, name_gesture)

                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_cfg()
    logger.info(OmegaConf.to_yaml(CFG))
    if CFG.mode == CFG.list_mode[0]:
        for mode in CFG.list_mode[1:]:
            unittest.main(argv=['ignored', '-v', f'MyCase.{mode}'], exit=False)    
    elif CFG.mode in CFG.list_mode[1:]:
        unittest.main(argv=['ignored', '-v', f'MyCase.{CFG.mode}'], exit=False)
    else:
        logger.info(f'mode must be in {CFG.list_mode}')
