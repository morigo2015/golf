from collections import deque
import numpy as np
import math
import cv2 as cv
from util import Util

debug = True


class ShotFinder:
    BALL_AVG_AREA = 2000  # 2000 # примерно площадь мяча (на среднем расстоянии)
    FRAMES_BEFORE_IMPACT = 100
    FRAMES_AFTER_IMPACT = 50

    ball_cont = None
    ball_center_lst = []
    impact_cnt = None
    state = 'shot_complete'  # 'shot_complete' -> 'ball_found' -> 'impact_started' -> 'shot_complete'

    # поток кадров --> shot_descriptor (frame_lst, impact_cnt, ball_spot)
    @classmethod
    def next_frame(cls, frame, frame_cnt):  # None or shot_descr
        FrameStore.store_frame(frame, frame_cnt)
        if debug:
            print(f"{frame_cnt=} {cls.state=}")

        if cls.state == 'shot_complete':
            cls.ball_cont = cls.find_still_ball(frame, frame_cnt)
            if cls.ball_cont is not None:
                cls.state = 'ball_found'
            return None

        if cls.state == 'ball_found':
            cls.impact_cnt = cls.ball_removed(frame, frame_cnt)
            if cls.impact_cnt:
                print(f"Impact found!! at frame {cls.impact_cnt}")
                cls.state = 'impact_started'
            return None

        if cls.state == 'impact_started':
            if frame_cnt != cls.impact_cnt + cls.FRAMES_AFTER_IMPACT:  # impact found but shot is not finished yet
                return None
            # last frame of the shot
            shot_desc = (FrameStore.restore_last_frames(cls.FRAMES_BEFORE_IMPACT + cls.FRAMES_AFTER_IMPACT),
                         cls.FRAMES_BEFORE_IMPACT)
            cls.impact_cnt = None
            cls.state = 'shot_complete'
            return shot_desc

    @staticmethod
    def ball_removed(frame, frame_cnt):
        # return frame_cnt if ball removed from his place else None
        return 1  # None

    @classmethod
    def find_still_ball(cls, frame, frame_cnt):
        # return still ball contour and set ball_cont else None
        mask = BgSubtractor.next_frame(frame)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # cv.imshow(f"mask", mask)
        ball_contour = cls.get_ball_cont(mask)
        if ball_contour is None:
            return None
        # ball found.

        # check if all balls in list are on the same place for 5 frames
        M = cv.moments(ball_contour)
        ball_center = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        cls.ball_center_lst.append(ball_center)
        if max(np.std(cls.ball_center_lst, axis=0) > 5):
            # deviate more than 5 pixels
            cls.ball_center_lst = []  # drop the list of balls
            return None
        # all balls in list are on the same place

        if len(cls.ball_center_lst) < 5:
            return None
        # ball is on the same place long enough

        # r = ((cv.contourArea(ball_contour)) / math.pi) ** 0.5
        cls.ball_cont = ball_contour
        cls.ball_center_lst = []
        if debug:
            print(f"still ball found {frame_cnt=}")  # area = {cv.contourArea(ball_contour)}
            x, y, w, h = (cv.boundingRect(cls.ball_cont))
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            Util.show_img(frame, f"still ball info {frame_cnt=}")
        return ball_contour

    @classmethod
    def get_ball_cont(cls, mask):
        # if mask contains a ball return it's contour else None
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ball_cont_lst = [c for c in contours if cls.BALL_AVG_AREA * 0.5 < cv.contourArea(c) < cls.BALL_AVG_AREA * 2]
        if len(ball_cont_lst) != 1:
            return None
        cont = ball_cont_lst[0]

        hull = cv.convexHull(cont)
        convex_ratio = cv.contourArea(cont) / cv.contourArea(hull)
        if not (convex_ratio > 0.95):
            # print(f"not convex contour. ratio={convex_ratio}")
            return None  # non-convex contour
        return cont


class BgSubtractor:
    background = None
    cv_backSub = None

    @classmethod
    def next_frame(cls, frame):
        if cls.cv_backSub is None:
            cls.cv_backSub = cv.createBackgroundSubtractorMOG2(varThreshold=140)
        return cls.cv_backSub.apply(frame)

        # if cls.background is None:  # first frame - init bg
        #     cls.background = cls.blur_image(frame)
        #
        # blur_img = cls.blur_image(frame)
        # diff = cv.absdiff(cls.background, blur_img)
        # mask = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)[1]
        # return mask
    #
    # @staticmethod
    # def blur_image(img):
    #     # img = cv.medianBlur(img, 3)
    #     img = cv.blur(img, (3, 3))
    #     gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    #     # gray = cv.blur(gray, (3,3))
    #     # gray = cv.medianBlur(gray, 3)
    #     return gray


class FrameStore:
    @classmethod
    def store_frame(cls, frame, frame_cnt):
        pass

    @classmethod
    def restore_last_frames(cls, frames_qty):
        # return last frames_qty frames and clear storage
        pass
