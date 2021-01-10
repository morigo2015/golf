import logging
from collections import deque

import cv2 as cv
import numpy as np

from player import Player

inp_file = 'video/tst/tst-balls-3.avi'


class SeekFirst:
    BALL_MIN_AREA, BALL_MAX_AREA = 1000, 4000

    def __init__(self, frame, frame_cnt, frame_name):
        self.rb = RingBuffer(60)
        self.bg_gray = blur_gray(frame)

        self.circle_mask = cv.circle(np.zeros((60, 60), np.uint8), (30, 30), 25, 255, -1)
        print(
            f"r=30 {cv.contourArea(cv.findContours(self.circle_mask,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0])=}")

    def next_frame(self, frame, frame_cnt, frame_name):
        # logging.debug(f"{frame_cnt=} {frame_name=} ")

        diff = cv.absdiff(self.bg_gray, blur_gray(frame))
        thresh = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)[1]
        cur_mask = cv.morphologyEx(thresh, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))
        self.rb.add(cur_mask)
        cv.imshow("current mask", cur_mask)

        stacked_mask = cur_mask
        for prev_mask in self.rb.get_lst():
            stacked_mask = cv.bitwise_and(stacked_mask, prev_mask, mask=cur_mask)
        cv.imshow("stacked mask", stacked_mask)

        contours, _ = cv.findContours(stacked_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ball_cont_lst = [c for c in contours if self.BALL_MIN_AREA < cv.contourArea(c) < self.BALL_MAX_AREA]

        ball_cont_lst = [c for c in ball_cont_lst
                         if (cv.contourArea(c) / cv.contourArea(cv.convexHull(c))) > 0.95]

        if len(ball_cont_lst) != 1:
            return frame, None
        ball_cont = ball_cont_lst[0]

        match_rate = cv.matchShapes(ball_cont, self.circle_mask, 1, 0.0)
        print(f"{match_rate=}")
        if not (0.85 < match_rate < 1.15):
            logging.debug(f"cont dropped since match_rate = {match_rate}")
            return frame, None

        # ball start position is found
        logging.debug(f"Ball found {frame_cnt}. {cv.contourArea(ball_cont)=:.0f} "
                      f"{(cv.contourArea(ball_cont) / cv.contourArea(cv.convexHull(ball_cont)))=:.2f} ")
        cv.drawContours(frame, [ball_cont], -1, (255, 0, 0), 3)

        ball_desc = ball_cont
        return frame, ball_desc


class SeekRepeatBall:

    def __init__(self, frame, bg_gray, ball_desc):
        self.bg_gray = bg_gray
        self.seek_mask, ball_area, self.first_ball_cont = self.seek_param(frame, ball_desc)
        self.ball_min_area, self.ball_max_area = ball_area * 0.7, ball_area * 1.6
        self.rb = RingBuffer(5)

    def next_frame(self, frame, frame_cnt, frame_name):
        diff = cv.absdiff(self.bg_gray, blur_gray(frame))
        thresh = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)[1]
        thresh = cv.bitwise_and(thresh, thresh, self.seek_mask)
        cur_mask = cv.morphologyEx(thresh, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))
        self.rb.add(cur_mask)
        cv.imshow("current mask", cur_mask)

        stacked_mask = cur_mask
        for prev_mask in self.rb.get_lst():
            stacked_mask = cv.bitwise_and(stacked_mask, prev_mask, mask=cur_mask)
        cv.imshow("stacked mask", stacked_mask)

        contours, _ = cv.findContours(stacked_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ball_cont_lst = [c for c in contours if self.ball_min_area < cv.contourArea(c) < self.ball_max_area]

        ball_cont_lst = [c for c in ball_cont_lst
                         if (cv.contourArea(c) / cv.contourArea(cv.convexHull(c))) > 0.95]  # или лучше с circle?

        ball_cont_lst = [c for c in ball_cont_lst
                         if cv.matchShapes(c, self.first_ball_cont, 1, 0.0) < 0.15]

        if len(ball_cont_lst) != 1:
            return frame, None

        ball_cont = ball_cont_lst[0]
        # ball start position is found
        logging.debug(f"Repeat Ball found {frame_cnt}. {cv.contourArea(ball_cont)=:.0f} "
                      f"{(cv.contourArea(ball_cont) / cv.contourArea(cv.convexHull(ball_cont)))=:.2f} ")
        cv.drawContours(frame, [ball_cont], -1, (255, 255, 0), 3)

        return frame, ball_cont

    @staticmethod
    def seek_param(frame, ball_desc):
        ball_contour = ball_desc
        # m = cv.moments(ball_contour)
        # ball_center = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
        x, y, w, h = (cv.boundingRect(ball_contour))
        rect_p1 = max(0, x - 2 * w), max(0, y - 2 * h)
        rect_p2 = min(1920, x + 3 * w), min(1080, y + 3 * h)
        seek_mask = np.zeros(frame.shape[0:2], np.uint8)
        seek_mask = cv.rectangle(seek_mask, rect_p1, rect_p2, 255, cv.FILLED)

        ball_area = cv.contourArea(ball_contour)
        return seek_mask, ball_area, ball_contour


class CheckRemoveBall:
    def __init__(self, ball_desc):
        self.ball_desc = ball_desc
        pass

    def next_frame(self, frame, frame_cnt, frame_name):
        return frame


class BallWatch:
    first_search, rpt_search, chk_remove = None, None, None
    cur_state = 'start'  # start -> search_first_ball -> search_repeat_ball -> search_repeat_ball -> ....
    idle_frames = None

    @classmethod
    def next_frame(cls, frame, frame_cnt, frame_name):
        if frame_cnt < 5:
            return frame
        logging.debug(f"BallWatch.next_frame: {frame_cnt=} {cls.cur_state=}")

        if cls.cur_state == 'start':  # first_search frame only
            cls.first_search = SeekFirst(frame, frame_cnt, frame_name)
            cls.cur_state = 'search_first'
            return frame

        elif cls.cur_state == 'search_first':
            frame, cls.ball_desc = cls.first_search.next_frame(frame, frame_cnt, frame_name)
            if cls.ball_desc is None:
                return frame
            cv.imshow(f"Ball found at {frame_cnt}", frame)
            cls.rpt_search = SeekRepeatBall(frame, cls.first_search.bg_gray, cls.ball_desc)
            cls.cur_state = 'search_repeat'
            return frame

        elif cls.cur_state == 'search_repeat':
            assert (cls.rpt_search is not None)
            frame, cls.ball_desc = cls.rpt_search.next_frame(frame, frame_cnt, frame_name)
            if cls.ball_desc is None:
                return frame
            # ball repeatedly found
            cv.imshow(f"Ball found at {frame_cnt}", frame)
            cls.chk_remove = CheckRemoveBall(cls.ball_desc)
            cls.cur_state = 'check_remove'
            return frame

        elif cls.cur_state == 'check_remove':
            assert (cls.chk_remove is not None)
            if cls.idle_frames is None:
                cls.idle_frames = 200
                cls.cur_state = 'check_remove'
            cls.idle_frames -= 1
            if not cls.idle_frames:
                cls.idle_frames = None
                cls.cur_state = 'search_repeat'  # debug
            return frame


def blur_gray(frame):
    frame = cv.medianBlur(frame, 3)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame


class RingBuffer:
    # not optimized (list instead of deque) ring buffer
    max_len = None
    items_lst = []

    def __init__(self, max_length):
        self.max_len = max_length

    def add(self, item):
        self.items_lst.append(item)
        if len(self.items_lst) > self.max_len:
            del self.items_lst[0]  # to be optimized later

    def get_lst(self):
        return self.items_lst



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
    Player.frame_processor = BallWatch.next_frame
    Player.player()


if __name__ == '__main__':
    main()

# if cls.bg_sub is None:
#     cls.bg_sub = cv.createBackgroundSubtractorMOG2() # varThreshold=140)
# bg_mask = cls.bg_sub.apply(frame)
# bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_DILATE, np.ones((7, 7), np.uint8))

# cv.imshow("bg_mask", bg_mask)
