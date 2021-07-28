# frame_processor for cutter (cut stream to short swing clips)
import logging
import re
from collections import deque
import datetime

import cv2 as cv
import numpy as np

from zone import OnePointZone

SWING_CLIP_PREFIX = "video/swings/"
NEED_TRANSPOSE = True
# need_transpose = False
INPUT_SCALE = 0.7

debug_flg = True
BLUR_LEVEL = 3

ROI_CENTER = int(30 * INPUT_SCALE)


class StartArea:
    thresh_val = None
    contour = None
    x, y, w, h = None, None, None, None
    ball_area = None


def find_best_threshold(gray):
    level_results = []
    for thresh in range(50, 200, 5):
        _, img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:  # no contours which include click_xy or too much
            continue
        cont_lst = [cont for cont in contours if
                    cv.pointPolygonTest(cont, (ROI_CENTER, ROI_CENTER), measureDist=False) >= 0]
        if len(cont_lst) != 1:  # no contours which include click_xy
            continue
        cont = cont_lst[0]  # it should be the one only contour which include click_xy
        area = cv.contourArea(cont)
        x, y, w, h = cv.boundingRect(cont)
        if max(w, h) == 2 * ROI_CENTER:
            continue
        logging.debug(f"{thresh=}: {area=} {x=} {y=} {w=} {h=}")
        # Util.show_img(img, "thresh find", 0)
        level_results.append({"thresh": thresh, "area": area, "d": max(w, h)})
    thresh_otsu, _ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    level_results = sorted(level_results, key=lambda result: result["area"], reverse=True)
    logging.debug(f"{len(level_results)=} {thresh_otsu=}")
    if len(level_results) > 1:  # return second best by area if possible
        return level_results[1]["thresh"]
    if len(level_results) == 1:  # return just best if there is only one result
        return level_results[0]["thresh"]
    return None


def get_start_area(frame):
    global ROI_CENTER
    if not OnePointZone.zone_is_defined():
        return
    xc, yc = OnePointZone.zone_point

    x_max, y_max = frame.shape[1], frame.shape[0]
    roi_x, roi_y = max(xc - ROI_CENTER, 0), max(yc - ROI_CENTER, 0)
    roi_w, roi_h = min(x_max - roi_x, ROI_CENTER * 2), min(y_max - roi_y, ROI_CENTER * 2)
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (BLUR_LEVEL, BLUR_LEVEL), 0)
    kernel = np.ones((BLUR_LEVEL, BLUR_LEVEL), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    # Util.show_img(gray, "StartArea: gray", 1)

    # StartArea.thresh_val, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh = find_best_threshold(gray)
    if not thresh:  # can't found ball contour
        print(" can't found ball where clicked")
        logging.debug(" can't found ball where clicked")
        return
    StartArea.thresh_val, thresh_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)

    # Util.show_img(thresh_img, "StartArea: thresh_img", 0)

    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cont_lst = [cont for cont in contours if
                cv.pointPolygonTest(cont, (ROI_CENTER, ROI_CENTER), measureDist=False) >= 0]
    if len(cont_lst) == 1:
        x, y, w, h = cv.boundingRect(cont_lst[0])
        d = max(w, h)
        if d == 2 * ROI_CENTER:  # nothing worth is found, contour is for full image
            # thresh_val = find_best_param(gray)
            return
        StartArea.contour = cont_lst[0]
        StartArea.ball_area = cv.contourArea(StartArea.contour)

        StartArea.x, StartArea.y = roi_x + x - 2 * d, roi_y + y - 2 * d  # 2 cells aside
        StartArea.w, StartArea.h = 5 * d, 5 * d  # 2 cells + original cell + 2 cells
        logging.debug(f"StartArea set by ball position: {StartArea.x=},{StartArea.y=}   {StartArea.w=},{StartArea.h=}  {StartArea.ball_area=}")
    elif len(cont_lst) == 0:  # not found any contour around click_xy
        return
    elif len(cont_lst) > 1:
        logging.error(
            f"!!!! Error !!! several contours include one point. ROI_SHFT= {ROI_CENTER}, cont_lst= {cont_lst}")

    OnePointZone.reset_zone_point()
    return


def is_touched_border(contour):
    x, y, w, h = cv.boundingRect(contour)
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
FRAME_BUFF_SZ = 300
MAX_CLIP_SZ = 150
frames_buffer = deque(maxlen=FRAME_BUFF_SZ)


def write_swing_clip(r):
    global status_history, frames_buffer, SWING_CLIP_PREFIX
    start_pos, end_pos = r.span()
    frames_to_write = min(end_pos - start_pos, MAX_CLIP_SZ)
    frames_to_skip = len(frames_buffer) - frames_to_write
    for i in range(frames_to_skip):
        frames_buffer.popleft()
    out_file_name = f"{SWING_CLIP_PREFIX}{datetime.datetime.now().strftime('%H:%M:%S')}.avi"  # f"{swing_clip_prefix}{swing_clip_cnt}.avi"
    out = None
    for i in range(frames_to_write):
        out_frame = frames_buffer.popleft()
        if not out:
            out = cv.VideoWriter(out_file_name, cv.VideoWriter_fourcc(*'XVID'), 20.0, (out_frame.shape[1], out_frame.shape[0]))
        out.write(out_frame)
    out.release()
    logging.debug(f"swing clip written: {out_file_name=} {start_pos=} {end_pos=}")
    return out_file_name


def frame_processor(frame, frame_cnt):
    global status_history, frames_buffer
    if frame_cnt == 1:
        OnePointZone.reset_zone_point()  # it points to ball so we have to re-init it each time

    if INPUT_SCALE != 1.0:
        cv.resize(frame, None, fx=INPUT_SCALE, fy=INPUT_SCALE)  # !!!
    if NEED_TRANSPOSE:
        frame = cv.transpose(frame)
    frame = cv.flip(frame, 1)

    get_start_area(frame)
    if StartArea.x is None:
        return frame

    status = get_start_area_status(frame)
    status_history += status
    frames_buffer.append(frame.copy())
    logging.debug(f"{len(frames_buffer)=}")

    cv.rectangle(frame, (StartArea.x, StartArea.y), (StartArea.x + StartArea.w, StartArea.y + StartArea.h), (255, 0, 0), 1)
    # cv.drawContours(frame, [StartArea.contour], 0, (0, 0, 255), 3)
    cv.putText(frame, f"{status}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    r = re.search('B{7}B*[MB]{0,7}E{15}$', status_history)  # B{7}[MB]*E{7}$
    logging.debug(f"{frame_cnt=}:  {status=}, {status_history=}")
    if r:
        out_file_name = write_swing_clip(r)
        print(f"hit!!!!  {frame_cnt=} {out_file_name=}")
        logging.debug(f"Hit: {r.string=}  {status_history=} {r.span()=}")
        status_history = ''
        frames_buffer.clear()

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
