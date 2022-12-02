import json
import numpy as np
import cv2
import math


def draw_gesture(img: np.array, name_gesture: str) -> np.array:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    org = (50, 50)
    thickness = 2
    fontScale = 0.5
    img = cv2.putText(img, name_gesture, org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return img


with open('asserts/hand_pose.json', 'r') as f:
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


class preprocessdata:

    def __init__(self, topology, num_parts):
        self.joints = []
        self.dist_bn_joints = []
        self.topology = topology
        self.num_parts = num_parts

    def svm_accuracy(self, test_predicted, labels_test):
        """"
        This method calculates the accuracy of the model 
        Input: test_predicted - predicted test classes
               labels_test
        Output: accuracy - of the model 
        """
        predicted = []
        for i in range(len(labels_test)):
            if labels_test[i] == test_predicted[i]:
                predicted.append(0)
            else:
                predicted.append(1)
        accuracy = 1 - sum(predicted)/len(labels_test)
        return accuracy

    def trainsvm(self, clf, train_data, test_data, labels_train, labels_test):
        """
        This method trains the different gestures 
        Input: clf - Sk-learn model pipeline to train, You can choose an SVM, linear regression, etc
                train_data - preprocessed training image data -in this case the distance between the joints
                test_data - preprocessed testing image data -in this case the distance between the joints
                labels_train - labels for training images 
                labels_test - labels for testing images 
        Output: trained model, predicted_test_classes
        """
        clf.fit(train_data, labels_train)
        predicted_test = clf.predict(test_data)
        return clf, predicted_test
    # def loadsvmweights():

    def joints_inference(self, image_shape, counts, objects, peaks):
        """
        This method returns predicted joints from an image/frame
        Input: image, counts, objects, peaks
        Output: predicted joints
        """
        joints_t = []
        height = image_shape[0]
        width = image_shape[1]
        K = self.topology.shape[0]
        count = int(counts[0])
        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                picked_peaks = peaks[0][j][k]
                joints_t.append(
                    [round(float(picked_peaks[1]) * width), round(float(picked_peaks[0]) * height)])
        joints_pt = joints_t[:self.num_parts]
        rest_of_joints_t = joints_t[self.num_parts:]

        # when it does not predict a particular joint in the same association it will try to find it in a different association
        for i in range(len(rest_of_joints_t)):
            l = i % self.num_parts
            if joints_pt[l] == [0, 0]:
                joints_pt[l] = rest_of_joints_t[i]

        # if nothing is predicted
        if count == 0:
            joints_pt = [[0, 0]]*self.num_parts
        return joints_pt

    def find_distance(self, joints):
        """
        This method finds the distance between each joints 
        Input: a list that contains the [x,y] positions of the 21 joints 
        Output: a list that contains the distance between the joints 
        """
        joints_features = []
        for i in joints:
            for j in joints:
                dist_between_i_j = math.sqrt((i[0]-j[0])**2+(i[1]-j[1])**2)
                joints_features.append(dist_between_i_j)
        return joints_features
