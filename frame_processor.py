# frame_processor for input stream

import cv2 as cv
import numpy as np

from util import Util

need_transpose = True
delay = 1

def get_white_mask_hsv(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    sensitivity = 150
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    # Threshold the HSV image to get only white colors
    white_mask = cv.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    # white_masked = cv.bitwise_and(frame, frame, mask=white_mask)
    Util.show_img(white_mask, "white mask", delay)
    return white_mask


def get_white_mask_hls(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    sensitivity = 100
    lower_white = np.array([0, 255-sensitivity, 0])
    upper_white = np.array([255, 255, 255])
    # Threshold the HSV image to get only white colors
    white_mask = cv.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    # white_masked = cv.bitwise_and(frame, frame, mask=white_mask)
    Util.show_img(white_mask, "white mask hsl", delay)
    return white_mask


def frame_processor(frame, frame_cnt):
    print(f"frame_processor, cnt={frame_cnt}")
    frame = cv.transpose(frame)
    frame = cv.flip(frame, 1)
    white_mask = get_white_mask_hls(frame)
    return frame

def end_stream_processor(frame_cnt):
    print(f"end of stream. frame_cnt={frame_cnt}")
    pass
