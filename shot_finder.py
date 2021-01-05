from collections import deque
import numpy as np
import cv2 as cv


class BgSubtractor:
    BG_HISTORY_LENGTH = 30 # 15  # how many frames a ball has to be found and still for detection
    background = None
    mask_history = deque([])

    cv_backSub = None

    @classmethod
    def next_frame(cls, frame):
        # if cls.cv_backSub is None:
        #     cls.cv_backSub = cv.createBackgroundSubtractorMOG2(varThreshold=140)
        # return cls.cv_backSub.apply(frame)

        if cls.background is None:  # first frame - init bg
            cls.background = cls.blur_image(frame)

        blur_img = cls.blur_image(frame)
        diff = cv.absdiff(cls.background, blur_img)
        mask = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)[1]
        # mask = cv.dilate(mask, None, iterations=2)
        cls.mask_history.append(mask)

        if len(cls.mask_history) < cls.BG_HISTORY_LENGTH:  # too few frames in history to make decision
            return None
        else:  # history is long enough
            # stock all masks together and look through them
            cls.mask_history.popleft()
            # and_mask = cls.mask_history[-1]
            or_mask = cls.mask_history[-1]
            # for i in range(len(cls.mask_history)-1):
            #     xor_mask = cv.bitwise_xor(cls.mask_history[i], cls.mask_history[i+1])
            #     cv.imshow("xor mask", xor_mask)
            #     or_mask = cv.bitwise_or(or_mask,xor_mask)
            return or_mask

    @staticmethod
    def blur_image(img):
        # img = cv.medianBlur(img, 3)
        img = cv.blur(img, (3,3))
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # gray = cv.blur(gray, (3,3))
        # gray = cv.medianBlur(gray, 3)
        return gray


class BallWatch:
    # prev = None
    @classmethod
    def impact_found(cls, frame, frame_cnt):
        # return True  # if impact found (ball removed from ready position)
        mask = BgSubtractor.next_frame(frame)
        if mask is None:
            return
        print(f"{frame_cnt=} {mask.mean()}")
        cv.imshow(f"mask", mask)
        # if cls.prev is None:
        #     cls.prev = mask.copy()
        # else:
            # xor_mask = cv.bitwise_xor(cls.prev,mask)
            # cv.imshow("xor_mask",cls.prev)
            # cls.prev = mask.copy()




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
