# frame_processor for cutter (cut stream to short swing clips)
import logging
import re

import cv2 as cv
import numpy as np

from zone import OnePointZone
from util import Util

need_transpose = True
delay = 1
debug_flg = True
BLUR_LEVEL = 3


class StartArea:
    thresh_val = None
    contour = None
    x, y, w, h = None, None, None, None
    ball_area = None


def get_start_area_data(frame):
    if not OnePointZone.zone_is_defined():
        return
    xc, yc = OnePointZone.zone_point

    ROI_SHFT = 20
    x_max, y_max = frame.shape[1], frame.shape[0]
    roi_x, roi_y = max(xc - ROI_SHFT, 0), max(yc - ROI_SHFT, 0)
    roi_w, roi_h = min(x_max - roi_x, ROI_SHFT * 2), min(y_max - roi_y, ROI_SHFT * 2)
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (BLUR_LEVEL, BLUR_LEVEL), 0)
    kernel = np.ones((BLUR_LEVEL, BLUR_LEVEL), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)

    StartArea.thresh_val, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cont_lst = [cont for cont in contours if cv.pointPolygonTest(cont, (ROI_SHFT, ROI_SHFT), measureDist=False) >= 0]
    if len(cont_lst) == 1:
        StartArea.contour = cont_lst[0]
        StartArea.ball_area = cv.contourArea(StartArea.contour)
        x, y, w, h = cv.boundingRect(StartArea.contour)
        d = max(w, h)
        StartArea.x, StartArea.y = roi_x + x - 2 * d, roi_y + y - 2 * d # 2 cells aside
        StartArea.w, StartArea.h = 5 * d, 5 * d  # 2 cells + original cell + 2 cells
        if debug_flg:
            logging.debug(f"StartArea set by ball position: \
                xy=({StartArea.x},{StartArea.y}) wh=({StartArea.w},{StartArea.h})  {StartArea.ball_area=}")
    elif len(cont_lst) == 0:  # not found any contour around click_xy, use ROI_SHFT as default
        return
    elif len(cont_lst) > 1:
        logging.error(f"!!!! Error !!! several contours include one point. ROI_SHFT= {ROI_SHFT}, cont_lst= {cont_lst}")

    OnePointZone.reset_zone_point()
    Util.show_img(thresh_img, "StartArea: thresh_img", 1)
    return True


def is_touched_border(contour):
    x, y, w, h = cv.boundingRect(StartArea.contour)
    return True \
        if x == StartArea.x or y == StartArea.y or x + w == StartArea.x + StartArea.w or y + h == StartArea.y + StartArea.h \
        else False


def get_start_area_status(frame):
    roi = frame[StartArea.y:StartArea.y + StartArea.h, StartArea.x:StartArea.x + StartArea.w]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (BLUR_LEVEL, BLUR_LEVEL), 0)
    kernel = np.ones((BLUR_LEVEL, BLUR_LEVEL), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    # Util.show_img(gray, "gray", 1)

    _, thresh_img = cv.threshold(gray, StartArea.thresh_val, 255, cv.THRESH_BINARY)
    # Util.show_img(thresh_img, "Stream:   thresh_img", 1)

    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if StartArea.ball_area * 0.5 < cv.contourArea(cnt)]
    all_cnt = len(contours)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) < StartArea.ball_area * 2]
    logging.debug(f"get_status: found contours: all = {all_cnt} not_big = {len(contours)} ")

    if all_cnt == 0:
        return 'E'
    if len(contours) == 1 and not is_touched_border(contours[0]):
        match_rate = cv.matchShapes(contours[0], StartArea.contour, 1, 0)
        logging.debug(f" {match_rate=}")
        if match_rate < 0.5:
            return 'B'
    return 'M'


status_history = ''

def frame_processor(frame, frame_cnt):
    global status_history
    if frame_cnt == 1:
        OnePointZone.reset_zone_point()  # it points to ball so we have to re-init it each time

    frame = cv.resize(frame, None, fx=0.5, fy=0.5)  # !!!
    # frame = cv.transpose(frame)
    frame = cv.flip(frame, 1)

    get_start_area_data(frame)
    if StartArea.x is None:
        return frame

    cv.rectangle(frame, (StartArea.x, StartArea.y), (StartArea.x + StartArea.w, StartArea.y + StartArea.h),
                 (255, 0, 0), 1)
    cv.drawContours(frame, [StartArea.contour], 0, (0, 0, 255), 3)

    status = get_start_area_status(frame)

    cv.putText(frame, f"{status}", (200, 200),
               cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)

    status_history += status
    r = re.search('B{5}M*E{5}', status_history)
    logging.debug(f"{frame_cnt}:  {status=}, {status_history=}")
    if r:
        print(f"hit!! {frame_cnt=} {status_history=} {r.span()} {r.string}")
        logging.debug(f"hit!! {frame_cnt=} {status_history=} {r.span()} {r.string}")
        status_history = ''

    return frame


def end_stream_processor(frame_cnt):
    print(f"end of stream. frame_cnt={frame_cnt}")
    pass


# ------------------------ old

'''
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
'''

"""
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
"""
