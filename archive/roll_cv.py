import logging
import cv2 as cv
import numpy as np

from player import Player
from ball_cv import RingBuffer
from util import Util

inp_file = 'video/tst/tst-roll-1.avi'

ROLL_BUF_LEN = 7
BALL_MIN_AREA, BALL_MAX_AREA = 700, 8000  # 1000, 4000


class SeekRoll:
    bg_sub = None
    rb = None

    @classmethod
    def next_frame(cls, frame, frame_name, frame_cnt):
        if cls.bg_sub is None:
            cls.bg_sub = cv.createBackgroundSubtractorMOG2(varThreshold=30)
            cls.rb = RingBuffer(ROLL_BUF_LEN)
        cur_mask = cls.bg_sub.apply(frame)
        cur_mask = cv.morphologyEx(cur_mask, cv.MORPH_ERODE, np.ones((7, 7), np.uint8), iterations=1)
        cur_mask = cv.morphologyEx(cur_mask, cv.MORPH_DILATE, np.ones((7, 7), np.uint8), iterations=3)
        cls.rb.add(cur_mask)
        # cv.imshow("current mask", cur_mask)

        stacked_or_mask = cur_mask
        stacked_and_mask = cur_mask
        for prev_mask in cls.rb.get_lst():
            stacked_and_mask = cv.bitwise_and(stacked_and_mask, prev_mask)
            stacked_or_mask = cv.bitwise_or(stacked_or_mask, prev_mask)
        neg_and = cv.bitwise_not(stacked_and_mask)
        or_neg_and = cv.bitwise_and(stacked_or_mask, neg_and)
        # cv.imshow("stacked AND mask", stacked_and_mask)
        # cv.imshow("stacked OR mask", stacked_or_mask)
        cv.imshow("stacked OR and (- AND) mask", or_neg_and)
        res_mask = or_neg_and

        contours, _ = cv.findContours(res_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        ball_cont_lst = [c for c in contours if
                         BALL_MIN_AREA * ROLL_BUF_LEN < cv.contourArea(c) < BALL_MAX_AREA * ROLL_BUF_LEN]
        rect_cont_lst = [(cv.minAreaRect(c), c) for c in ball_cont_lst]
        rect_cont_lst = [rc for rc in rect_cont_lst
                         if rc[0][0][1] < int(1080.*0.7)]  # exclude lower third part of frame (change later)
        rect_cont_lst = [rc for rc in rect_cont_lst if
                         Util.rect_length(rc[0]) / Util.rect_width(rc[0]) > ROLL_BUF_LEN - 2]

        if len(rect_cont_lst) != 1:
            return frame

        for ind, rc in enumerate(rect_cont_lst):
            cv.drawContours(frame, [rc[1]], -1, (255, 255, 0), 1)
        # print(
        #     f"Tail found {frame_cnt=} {ind=} {cv.contourArea(rc[1]):.0f} rlength/width={Util.rect_length(rc[0]) / Util.rect_width(rc[0]):.1f}")
        # Util.show_img(frame, f"Tail found at {frame_cnt=} {ind=}")

        # slice the tail to frames
        # rect = rc[0]
        cont = rect_cont_lst[0][1]
        tail_mask = Util.cont_mask(cont, frame.shape[0:2])
        prev_mask_lst = cls.rb.get_lst()
        sliced_mask_lst = [cv.bitwise_and(tail_mask, prev_mask) for prev_mask in prev_mask_lst]

        # sliced_mask_cont_lst = [cv.findContours(sm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        #                         for sm in sliced_mask_lst]  # !!! тут могут быть несколько контуров !!! исправить


        sliced_mask_centers_lst = [center_xy for smc in sliced_mask_lst if
                                   (center_xy := Util.center_xy(smc))[0] != -1]

        sliced_mask_shift_w_lst = [sliced_mask_centers_lst[i + 1][0] - sliced_mask_centers_lst[i][0]
                                   for i in range(len(sliced_mask_centers_lst) - 1)]
        if all((smsw > 0 for smsw in sliced_mask_shift_w_lst)):
            direction = 'left-2-right'
        elif all((smsw < 0 for smsw in sliced_mask_shift_w_lst)):
            direction = 'right-2-left'
        else:
            direction = 'chaotic'
            return frame




        print(f"Correct tail: {direction=} {sliced_mask_shift_w_lst=}")
        cv.imshow(f"Correct tail {frame_cnt=}",frame)
        return frame


def main():
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/debug.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    Player.inp_source_name = inp_file  # 'video/tst/tst-init-1.avi'
    Player.write_mode = False
    Player.frame_mode_initial = 0
    Player.frame_processor = SeekRoll.next_frame
    Player.player()


if __name__ == '__main__':
    main()
