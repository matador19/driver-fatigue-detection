import os
import sys
import argparse as argp
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import srt

import env
from console import Console
from crop_eyes import combined_cropped_eyes_image

DIR_PATH = os.path.dirname(__file__)
VIDEO_NAME = 'bara.mp4'
VIDEO_PATH = os.path.join(DIR_PATH, 'content', VIDEO_NAME)
VIDEO_SAMPLE_RATE = 30
SRT_NAME = 'bara.srt'
SRT_PATH = os.path.join(DIR_PATH, 'content', SRT_NAME)
SRT_ENCODING = 'utf-8-sig'

DATASET_DIR = os.path.join(DIR_PATH, 'dataset')


def get_classifier_classes(input_dict) -> set:
    s = set()
    for val in input_dict.values():
        s.add(val)
    return s


def parse_srt(path) -> dict:
    with open(path, encoding=SRT_ENCODING) as f:
        frame_class = {}
        for sub in srt.parse(f):
            s = int(timestamp_to_frame_num(sub.start))
            e = int(timestamp_to_frame_num(sub.end)) - 1
            for i in range(s, e):
                frame_class[i] = sub.content

        return frame_class


def parse_video(path, video_name, frames_dict):
    cap = cv2.VideoCapture(path)
    success = cap.read()  # get the next frame
    frame_num = 0
    while success:
    # for _ in range(5):
        frame_num += 1
        if 0 % VIDEO_SAMPLE_RATE == 0:
            success, img = cap.read()
            if img is not None:
                if frame_num in frames_dict:
                    frame_dir = frames_dict[frame_num]
                    frame_storage_path = os.path.join(DATASET_DIR, frame_dir, f'{video_name}_{frame_num}.jpg')
                    print(frame_storage_path)
                    cropped_image = combined_cropped_eyes_image(img)

                    if cropped_image is None:
                        continue

                    cv2.imwrite(frame_storage_path, cropped_image)


def timestamp_to_frame_num(timestamp):
    timedelta = pd.Timedelta(timestamp)
    return timedelta.total_seconds() * VIDEO_SAMPLE_RATE


def main():
    frames_class = parse_srt(SRT_PATH)
    classifier_classes = get_classifier_classes(frames_class)

    if not os.path.exists(DATASET_DIR):
        Console.info(logger="Directory structure", msg="Creating 'dataset' directory")
        os.mkdir(DATASET_DIR)

    for c in classifier_classes:
        c_dir_name = os.path.join(DATASET_DIR, c)
        if not os.path.exists(c_dir_name):
            Console.info(logger="Directory structure", msg=f'Creating \'dataset/{ c }\' directory')
            os.mkdir(c_dir_name)

    Console.info(logger="Dataset parser", msg="Beginning parsing of input video")

    video_name_without_ext = VIDEO_NAME.split('.')[0]
    parse_video(VIDEO_PATH, video_name_without_ext, frames_class)

    Console.info(logger="Dataset parser", msg="Video done parsing")


if __name__ == '__main__':
    main()
