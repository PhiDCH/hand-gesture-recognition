#!/usr/bin/python3

from torch_infer import TorchPose, SVMClassifier
from utils import draw_gesture, draw_joints
import cv2
from PIL import Image
import unittest
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
        self.assertIsInstance(TorchPose(CFG.model_pose), TorchPose)
        self.assertIsInstance(SVMClassifier(CFG.model_classify), SVMClassifier)

    def image(self,):
        model_pose = TorchPose(CFG.model_pose)
        img = cv2.imread(CFG.img_path)
        joints = model_pose(img.copy())
        img = draw_joints(model_pose.prepro(img), joints)

        model_classify = SVMClassifier(CFG.model_classify)
        name_gesture = model_classify(joints, single_res=True)
        img = draw_gesture(img, name_gesture)
        self.assertEqual(name_gesture, 'stop')
        if CFG.viz:
            Image.fromarray(img).show()

    def video(self,):
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
