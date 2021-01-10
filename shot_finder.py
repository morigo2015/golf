from collections import deque
import numpy as np
import math
import logging
import cv2 as cv
from util import Util

debug = True


class ShotFinder:
    BALL_AVG_AREA = 2000  # 2000 # примерно площадь мяча (на среднем расстоянии)
    FRAMES_BEFORE_IMPACT = 100
    FRAMES_AFTER_IMPACT = 50

    ball_desc = None
    ball_seek_param = None
    ball_center_lst = []
    impact_cnt = None
    state = 'shot_complete'  # 'shot_complete' -> 'is_ball_found' -> 'impact_started' -> 'shot_complete'
    background_subtr, bg_frame = None, None

    # поток кадров --> shot_descriptor (frame_lst, impact_cnt, ball_spot)
    @classmethod
    def next_frame(cls, frame, frame_cnt):  # None or shot_descr
        FrameStore.store_frame(frame, frame_cnt)
        logging.debug(f"{frame_cnt=} {cls.state=}")
        if frame_cnt == 1:  # init bg
            cls.background_subtr = cv.createBackgroundSubtractorMOG2(varThreshold=140)
            cls.bg_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        if cls.state == 'shot_complete':
            if cls.is_ball_found(frame, frame_cnt):
                if debug:
                    print(f"{frame_cnt=}  ball_foudnd set")
                cls.state = 'is_ball_found'
            return None

        if cls.state == 'is_ball_found':
            if cls.is_ball_removed(frame, frame_cnt):
                cls.impact_cnt = frame_cnt
                if debug:
                    print(f"{frame_cnt=}  Impact found!!")
                cls.state = 'impact_started'
            return None

        if cls.state == 'impact_started':
            if frame_cnt != cls.impact_cnt + cls.FRAMES_AFTER_IMPACT:  # impact found but shot is not finished yet
                return None
            # last frame of the shot
            shot_desc = (FrameStore.restore_last_frames(cls.FRAMES_BEFORE_IMPACT + cls.FRAMES_AFTER_IMPACT),
                         frame_cnt - cls.ball_desc[4], frame_cnt)
            if debug:
                print(f"************************** shot complete  {cls.ball_desc[4]} - {cls.impact_cnt} - {frame_cnt}")
                # Util.show_img(frame, f"{frame_cnt=} found ")
            cls.impact_cnt = None
            cls.state = 'shot_complete'
            return shot_desc

    @classmethod
    def is_ball_found(cls, frame, frame_cnt):
        # return still ball contour and set ball_desc else None

        if cls.ball_seek_param is None:  # first_search search
            fg_mask = cls.background_subtr.apply(frame)
            if cls.ball_seek_param:
                ball_seek_mask = cls.ball_seek_param[0]
                fg_mask = cv.bitwise_and(fg_mask, fg_mask, mask=ball_seek_mask)
                # cv.imshow("ball seek mask", ball_seek_mask)
            mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        else:  # not a first_search search, use ball_seek_param
            ball_seek_mask, area, ball_contour = cls.ball_seek_param
            img = cv.medianBlur(frame, 5)
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            # gray = cv.medianBlur(gray, 5)
            # diff = cv.absdiff(cls.start_area_bg, gray)
            thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)[1]
            thresh = cv.bitwise_and(thresh, thresh, mask=ball_seek_mask)
            mask = cv.dilate(thresh, None, iterations=2)
        if debug:
            cv.imshow(f"frame mask", mask)

        ball_contour = cls.get_ball_cont(mask)
        if ball_contour is None:
            return False
        # ball found.

        # check if all balls in list are on the same place for 5 frames
        M = cv.moments(ball_contour)
        ball_center = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        cls.ball_center_lst.append(ball_center)
        if max(np.std(cls.ball_center_lst, axis=0) > 5):
            # if debug:
            #     print(f"is_ball_found: drop list {cls.ball_center_lst}")
            # deviate more than 5 pixels
            cls.ball_center_lst = []  # drop the list of balls
            return False
        # all balls in list are on the same place

        if len(cls.ball_center_lst) < 4:
            return False
        # ball is on the same place long enough

        # r = ((cv.contourArea(ball_contour)) / math.pi) ** 0.5
        cls.ball_center_lst = []
        # x, y, w, h = (cv.boundingRect(ball_contour))

        ball_mask = np.zeros(frame.shape[0:2], np.uint8)
        ball_mask = cv.drawContours(ball_mask, [ball_contour], -1, 255, cv.FILLED)
        bg_ball_mean = cv.mean(cls.bg_frame, mask=ball_mask)
        found_ball_mean = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV), mask=ball_mask)
        found_ball_min, found_ball_max, _, _ = cv.minMaxLoc(mask, mask=ball_mask)
        cls.ball_desc = (ball_contour, ball_mask, bg_ball_mean[0], found_ball_mean[0],
                         frame_cnt, found_ball_min, found_ball_max)
        print(f"found_ball range: {found_ball_min} - {found_ball_max}")

        x, y, w, h = cv.boundingRect(ball_contour)
        if cls.ball_seek_param is None:  # it's first_search search, store info about start_area for future searches
            ball_seek_mask = np.zeros(frame.shape[0:2], np.uint8)
            p1 = max(0, x - 7 * w), max(0, y - 2 * h)
            p2 = min(1920, x + 8 * w), min(1080, y + 3 * h)
            ball_seek_mask = cv.rectangle(ball_seek_mask, p1, p2, 255, cv.FILLED)
            cls.ball_seek_param = (ball_seek_mask, cv.contourArea(ball_contour), ball_contour)
            frame_masked = cv.bitwise_and(frame, frame, mask=ball_seek_mask)
            # Util.show_img(frame_masked, "frame masked by ball seek mask")

        if debug:
            print(f"still ball found {frame_cnt=}, {bg_ball_mean=}")  # area = {cv.contourArea(ball_contour)}
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            masked_frame = cv.bitwise_and(frame, frame, mask=ball_mask)
            # Util.show_img(masked_frame, f"{frame_cnt=} still ball masked frame ")

        return True

    @classmethod
    def get_ball_cont(cls, mask):
        # if mask contains a ball return it's contour else None
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cls.ball_seek_param:
            min_area, max_area = cls.ball_seek_param[1] * 0.7, cls.ball_seek_param[1] * 1.3
        else:
            min_area, max_area = cls.BALL_AVG_AREA * 0.5, cls.BALL_AVG_AREA * 2
        ball_cont_lst = [c for c in contours if min_area < cv.contourArea(c) < max_area]
        if len(ball_cont_lst) != 1:
            return None
        cont = ball_cont_lst[0]

        hull = cv.convexHull(cont)
        convex_ratio = cv.contourArea(cont) / cv.contourArea(hull)
        if not (convex_ratio > 0.95):
            # print(f"not convex contour. ratio={convex_ratio}")
            return None  # non-convex contour

        if cls.ball_seek_param:
            first_ball_cont = cls.ball_seek_param[2]
            match_ratio = cv.matchShapes(cont,first_ball_cont,1,0.0)
            if not (match_ratio < 0.15):
                return None
            # matchs = [cv.matchShapes(cont,first_ball_cont,method,0.0) for method in [1,2,3]]
            # print(f" get ball match: {matchs[0]=:.02f} {matchs[1]=:.02f}{matchs[2]=:.02f}")

        return cont

    @classmethod
    def is_ball_removed(cls, frame, frame_cnt):
        # return frame_cnt if ball removed from his place else None
        ball_mask, bg_ball_mean, found_ball_mean = cls.ball_desc[1:4]
        cur_ball_mean = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV), mask=ball_mask)[0]
        # if debug:
        #     print(f"{frame_cnt=} is_ball_removed  ::: {bg_ball_mean=} {cur_ball_mean=} {found_ball_mean=}")
        if abs(cur_ball_mean - bg_ball_mean) < 10: # abs(cur_ball_mean - found_ball_mean):
            masked_frame = cv.bitwise_and(frame, frame, mask=ball_mask)
            # Util.show_img(masked_frame, f"{frame_cnt=} impact masked")
            return True
        else:
            return False


#
# class BgSubtractor:
#     background = None
#     cv_backSub = None
#
#     @classmethod
#     def next_frame(cls, frame):
#         if cls.cv_backSub is None:
#             cls.cv_backSub = cv.createBackgroundSubtractorMOG2(varThreshold=140)
#         return cls.cv_backSub.apply(frame)
#
#         # if cls.background is None:  # first_search frame - init bg
#         #     cls.background = cls.blur_image(frame)
#         #
#         # blur_img = cls.blur_image(frame)
#         # diff = cv.absdiff(cls.background, blur_img)
#         # mask = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)[1]
#         # return mask
#     #
#     # @staticmethod
#     # def blur_image(img):
#     #     # img = cv.medianBlur(img, 3)
#     #     img = cv.blur(img, (3, 3))
#     #     gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#     #     # gray = cv.blur(gray, (3,3))
#     #     # gray = cv.medianBlur(gray, 3)
#     #     return gray


class FrameStore:
    @classmethod
    def store_frame(cls, frame, frame_cnt):
        pass

    @classmethod
    def restore_last_frames(cls, frames_qty):
        # return last frames_qty frames and clear storage
        pass
