import torch
import torchvision
import tensorflow as tf
from mtcnn import MTCNN

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

vgg16 = torchvision.models.vgg16(pretrained=True)
size = 24
EYE_SIZE = 32

def extract_eye(point, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_img = img[point[1]-size:point[1] + size, point[0]-size:point[0] + size]
    return crop_img


def combined_cropped_eyes_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detections = detector.detect_faces(img)

    img_with_dets = img.copy()
    min_conf = 0.9
    keypoints = None

    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']
            keypoints = det['keypoints']
            cv2.rectangle(img_with_dets, (x, y), (x + width, y + height), (0, 155, 255), 2)
            cv2.circle(img_with_dets, (keypoints['left_eye']), 2, (0, 155, 255), 2)
            cv2.circle(img_with_dets, (keypoints['right_eye']), 2, (0, 155, 255), 2)
            cv2.circle(img_with_dets, (keypoints['nose']), 2, (0, 155, 255), 2)
            cv2.circle(img_with_dets, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
            cv2.circle(img_with_dets, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
    #  plt.figure(figsize=(10, 10))
    #  plt.imshow(img_with_dets)
    #  plt.axis('off')

    if keypoints is None:
        return None

    # get resized eye image
    r_eye_img_orig_size = extract_eye(keypoints['left_eye'], img)
    r_eye_img = cv2.resize(r_eye_img_orig_size, dsize=(EYE_SIZE, EYE_SIZE), interpolation=cv2.INTER_CUBIC)
    #  plt.imshow(r_eye_img)

    l_eye_img_orig_size = extract_eye(keypoints['right_eye'], img)
    l_eye_img = cv2.resize(l_eye_img_orig_size, dsize=(EYE_SIZE, EYE_SIZE), interpolation=cv2.INTER_CUBIC)
    #  plt.imshow(l_eye_img)

    combined_img = np.concatenate((r_eye_img, l_eye_img), axis=1)
    #  plt.imshow(combined_img)

    return combined_img
