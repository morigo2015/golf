import logging
from functools import reduce
import cv2 as cv
import numpy as np

from player import Player
from ball_cv import RingBuffer
from util import Util, FrameStream

inp_file = 'video/tst/tst-roll-1.avi'

ROLL_BUF_LEN = 7
BALL_MIN_AREA, BALL_MAX_AREA = 700, 8000  # 1000, 4000
TAIL_MIN_AREA, TAIL_MAX_AREA = BALL_MIN_AREA * ROLL_BUF_LEN, BALL_MAX_AREA * ROLL_BUF_LEN


class SeekRoll:
    bg_sub = None
    rb = None

    @classmethod
    def next_frame(cls, frame, frame_cnt, frame_name=""):
        # 0. init
        if cls.bg_sub is None:
            cls.bg_sub = cv.createBackgroundSubtractorMOG2(varThreshold=30)
            cls.rb = RingBuffer(ROLL_BUF_LEN)

        # build binary for current frame and store to ring buffer.
        cur_binary = cls.bg_sub.apply(frame)
        cur_binary = cv.morphologyEx(cur_binary, cv.MORPH_ERODE, np.ones((7, 7), np.uint8), iterations=1)
        cur_binary = cv.morphologyEx(cur_binary, cv.MORPH_DILATE, np.ones((7, 7), np.uint8), iterations=3)
        # cv.imshow("current mask", cur_binary)

        # store to ring buffer
        cls.rb.add(cur_binary)
        prev_lst = cls.rb.get_lst()
        # for ind,mask in enumerate(prev_lst):
        #     cv.imshow(f"prev mask {ind}",mask)

        # build prev_stacked: OR* and not(AND*)
        prev_OR = reduce(lambda x, y: cv.bitwise_or(x, y), prev_lst)
        prev_AND = reduce(lambda x, y: cv.bitwise_and(x, y), prev_lst)
        prev_AND_neg = cv.bitwise_not(prev_AND)
        prev_stacked = cv.bitwise_and(prev_OR, prev_AND_neg)
        cv.imshow("prev_stacked", prev_stacked)

        # filter contours in prev_stacked (area, length/width, similar to rotated rectangle) and build tail
        contours, _ = cv.findContours(prev_stacked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rect_conts = [(cv.minAreaRect(c), c) for c in contours if TAIL_MIN_AREA <= cv.contourArea(c) <= TAIL_MAX_AREA]
        rect_conts2 = [rc for rc in rect_conts if Util.rect_length(rc[0]) / Util.rect_width(rc[0]) > ROLL_BUF_LEN - 2]
        # rect_conts3 = [rc for rc in rect_conts2 if cv.contourArea(rc[1]) / Util.rect_area(rc[0]) > 0.8]
        rect_conts3 = rect_conts2

        if len(rect_conts3) != 1:
            # if len(rect_conts2)==0:
            #     return  frame
            # logging.debug(f"{frame_cnt=}: {'too many' if len(rect_conts3) > 1 else 'no'} contours after filter: "
            #               f"{len(contours)=} {len(rect_conts)=} {len(rect_conts3)=}")
            # cont = rect_conts2[0][1]
            # cv.drawContours(frame, [cont],-1,(0,0,255),1)
            # rect = rect_conts2[0][0]
            # cv.drawContours(frame, [np.int0(cv.boxPoints(rect))], 0, (0, 191, 255), 2)
            # cv.imshow( "rect_conts2[0]", frame)
            return frame  # позже можно добавить обработку если осталось 2 кандидата - выбираем ближнего к центру
        rect_cont = rect_conts3[0]
        tail = Util.cont_mask(rect_cont[1], frame.shape[0:2])
        cv.imshow("tail", tail)

        # build sliced_tail, centers, areas sliced_tail_lst (direction, area)
        sliced_tail = [cv.bitwise_and(tail, prev) for prev in prev_lst]
        # for ind,mask in enumerate(sliced_tail):
        #     cv.imshow(f"sliced tail {ind}",mask)
        # cv.waitKey(0)

        # check direction
        slice_centers = [Util.center_xy(slice) for slice in sliced_tail]  # return (-1,-1) if center can't be found
        if any((sc[0] == -1 for sc in slice_centers)):
            return frame  # skip frame if any slice in tail has no area (center_x = -1)
        # w_shifts = reduce(lambda sc_prev, sc: sc[0] - sc_prev[0], slice_centers)  # diff w for 2 seq centers
        w_shifts = [slice_centers[ind + 1][0] - slice_centers[ind][0] for ind in range(len(slice_centers) - 1)]
        if all((ws >= 0 for ws in w_shifts)):
            slice_direction = 'left-right'
        elif all((ws <= 0 for ws in w_shifts)):
            slice_direction = 'right-left'
        else:
            slice_direction = 'chaotic'
            # logging.debug(f"frame dropped since direction==chaotic. {w_shifts=}")
            return frame

        # check slice areas == tail_area/buf_len
        tail_area = cv.countNonZero(tail)
        slice_areas = [cv.countNonZero(slice) for slice in sliced_tail]
        if not (0.5 <= sum(slice_areas) / tail_area <= 2):
            # if any((not (ROLL_BUF_LEN - 2 <= tail_area / sa <= ROLL_BUF_LEN + 2) for sa in slice_areas)):
            # logging.debug(f"frame dropped since no approp contour area in tail slice. {slice_areas=}")
            return frame

        # for ind, rc in enumerate(rect_cont_lst):
        #     cv.drawContours(frame, [rc[1]], -1, (255, 255, 0), 1)
        # print(
        #     f"Tail found {frame_cnt=} {ind=} {cv.contourArea(rc[1]):.0f} rlength/width={Util.rect_length(rc[0]) / Util.rect_width(rc[0]):.1f}")
        # Util.show_img(frame, f"Tail found at {frame_cnt=} {ind=}")

        # slice the tail to frames
        # rect = rc[0]
        # cont = rect_cont_lst[0][1]
        # tail_mask = Util.cont_mask(cont, frame.shape[0:2])
        # prev_mask_lst = cls.rb.get_lst()
        # sliced_mask_lst = [cv.bitwise_and(tail_mask, prev_mask) for prev_mask in prev_mask_lst]

        # sliced_mask_cont_lst = [cv.findContours(sm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        #                         for sm in sliced_mask_lst]  # !!! тут могут быть несколько контуров !!! исправить

        # sliced_mask_centers_lst = [center_xy for smc in sliced_mask_lst if
        #                            (center_xy := Util.center_xy(smc))[0] != -1]
        #
        # sliced_mask_shift_w_lst = [sliced_mask_centers_lst[i + 1][0] - sliced_mask_centers_lst[i][0]
        #                            for i in range(len(sliced_mask_centers_lst) - 1)]
        # if all((smsw > 0 for smsw in sliced_mask_shift_w_lst)):
        #     direction = 'left-2-right'
        # elif all((smsw < 0 for smsw in sliced_mask_shift_w_lst)):
        #     direction = 'right-2-left'
        # else:
        #     direction = 'chaotic'
        #     return frame

        logging.debug(f"{frame_cnt=} Correct tail: {slice_direction=}")
        # cv.imshow(f"Correct tail {frame_cnt=}", frame)
        # Cutter.roll_found(frame_cnt)
        return frame


# class Cutter:
#     FRAMES_BEFORE_ROLL = 100
#     FRAMES_AFTER_ROLL = 100
#     roll_found_lst = []
#     segments = []
#
#     @classmethod
#     def roll_found(cls, frame_cnt):
#         if not len(cls.segments) or cls.segments[-1][2] < frame_cnt:
#             cls.segments.append((max(frame_cnt - cls.FRAMES_BEFORE_ROLL,1), frame_cnt, frame_cnt + cls.FRAMES_AFTER_ROLL))
#         else:
#             logging.debug(f"roll found at {frame_cnt=}. skipped since last cut is not ended yet")
#
#     @classmethod
#     def cut(cls, inp_fname):
#         fs = FrameStream(inp_fname)
#         found_iter = iter(cls.roll_found_lst)
#         next_start, next_found, next_stop = next(found_iter)
#         while True:
#             frame, frame_name, frame_cnt = fs.next_frame()
#             if frame_cnt == next_start:
#


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

    # print(f"roll found: {Cutter.roll_found_lst}")
    # Cutter.cut(inp_file)


if __name__ == '__main__':
    main()
