# frame_processor for hit statistics

import cv2 as cv
import numpy as np

from zone import ZonePoints
from util import Util

need_transpose = True
delay = -1


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
    sensitivity = 120
    lower_white = np.array([0, 255 - sensitivity, 0])
    upper_white = np.array([255, 255, 255])
    # Threshold the HSV image to get only white colors
    white_mask = cv.inRange(hsv, lower_white, upper_white)
    # Util.show_img(white_mask, "white mask hsl", delay)
    return white_mask

def get_controur_mask(frame):
    zone_contour = ZonePoints.get_zone_contour()
    zone_mask = np.zeros((frame.shape[0],frame.shape[1],1), np.uint8)
    if len(zone_contour):
        zone_mask = cv.drawContours(zone_mask, [zone_contour], -1, (255), thickness=cv.FILLED)
    # Util.show_img(zone_mask, "zone mask", delay)
    return zone_mask

def frame_processor(frame, frame_cnt):
    frame = cv.transpose(frame)
    frame = cv.flip(frame, 1)
    white_mask = get_white_mask_hls(frame)
    zone_mask = get_controur_mask(frame)
    zone_white = cv.bitwise_and(white_mask,zone_mask)
    Util.show_img(zone_white, "zone & white mask", delay)

    ball_cnts, _ = cv.findContours(zone_white,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ball_cnts = [ cnt for cnt in ball_cnts if cv.contourArea(cnt)>5]
    ball_cnts = sorted(ball_cnts, key=lambda cnt: cv.contourArea(cnt), reverse=True )
    if len(ball_cnts):
        print(f"{frame_cnt}: {[(cv.contourArea(cnt),cv.boundingRect(cnt)) for cnt in ball_cnts ]}")

    cv.drawContours(frame,ball_cnts,-1,(0,0,255),1)
    return frame


def end_stream_processor(frame_cnt):
    print(f"end of stream. frame_cnt={frame_cnt}")
    pass
