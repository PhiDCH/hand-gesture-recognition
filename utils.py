import numpy as np 
import cv2


def draw_gesture(img: np.array, name_gesture: str) -> np.array:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    org = (50, 50)
    thickness = 2
    fontScale = 0.5
    img = cv2.putText(img, name_gesture, org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return img

import json
with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)
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