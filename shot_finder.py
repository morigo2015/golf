from collections import deque
import numpy as np
import math
import cv2 as cv
from util import Util


class BgSubtractor:
    background = None
    cv_backSub = None

    @classmethod
    def next_frame(cls, frame):
        if cls.cv_backSub is None:
            cls.cv_backSub = cv.createBackgroundSubtractorMOG2(varThreshold=140)
        return cls.cv_backSub.apply(frame)

        if cls.background is None:  # first frame - init bg
            cls.background = cls.blur_image(frame)

        blur_img = cls.blur_image(frame)
        diff = cv.absdiff(cls.background, blur_img)
        mask = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)[1]
        return mask

    @staticmethod
    def blur_image(img):
        # img = cv.medianBlur(img, 3)
        img = cv.blur(img, (3, 3))
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # gray = cv.blur(gray, (3,3))
        # gray = cv.medianBlur(gray, 3)
        return gray


class BallWatch:
    BALL_AVG_AREA = 2000  # 2000 # примерно площадь мяча (на среднем расстоянии)

    calibrate_info = None
    calibrate_center_lst = []

    @classmethod
    def impact_found(cls, frame, frame_cnt):
        # return True  # if impact found (ball removed from ready position)
        cls.calibrate(frame, frame_cnt)

        # if cls.calibrate_info is None:
        #     cls.calibrate(frame, frame_cnt)
        #     return False


        # print(f"ball starting area found at: {frame_cnt}")
        # cv.waitKey(0)
        # exit(0)

    @classmethod
    def calibrate(cls, frame, frame_cnt):
        cls.calibrate_info = None # ***********************
        mask = BgSubtractor.next_frame(frame)
        # if mask is None:
        #     return
        # print(f"{frame_cnt=} {mask.mean()}")
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5,5),np.uint8))
        cv.imshow(f"mask", mask)
        ball_contour = cls.get_ball(mask)
        if ball_contour is None:
            return
        M = cv.moments(ball_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cls.calibrate_center_lst.append((cx,cy))
        calibr_cx, calibr_cy = np.mean(cls.calibrate_center_lst, axis=0)
        if abs(calibr_cx - cx)>5 or abs(calibr_cy-cy)>5:
            cls.calibrate_center_lst=[]
            return
        if len(cls.calibrate_center_lst)==4:
            # r = ((cv.contourArea(ball_contour)) / math.pi) ** 0.5
            cls.calibrate_info = ball_contour  # add more info for ball area here
            print(f"calibrate. {frame_cnt=}") # area = {cv.contourArea(ball_contour)}
            x, y, w, h = (cv.boundingRect(cls.calibrate_info))
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow(f"calibrate info {frame_cnt=}",frame)

    @classmethod
    def get_ball(cls, mask):
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


class FrameStore:
    @classmethod
    def store_frame(cls, frame, frame_cnt):
        pass

    @classmethod
    def restore_last_frames(cls, frames_qty):
        # return last frames_qty frames and clear storage
        pass


class ShotFinder:
    FRAMES_BEFORE_IMPACT = 100
    FRAMES_AFTER_IMPACT = 50
    impact_cnt = None

    # поток кадров --> shot_descriptor (frame_lst, impact_cnt, ball_spot)
    @classmethod
    def next_frame(cls, frame, frame_cnt):  # None or shot_descr
        if BallWatch.impact_found(frame, frame_cnt):
            print(f"Impact found!! at frame {frame_cnt}")
            cls.impact_cnt = frame_cnt
        if cls.impact_cnt is None:  # impact not found yet
            return None
        if frame_cnt != cls.impact_cnt + cls.FRAMES_AFTER_IMPACT:  # impact found but shot is not finished yet
            return None
        # last frame of the shot
        shot_desc = (FrameStore.restore_last_frames(cls.FRAMES_BEFORE_IMPACT + cls.FRAMES_AFTER_IMPACT),
                     cls.FRAMES_BEFORE_IMPACT)
        cls.impact_cnt = None
        return shot_desc
