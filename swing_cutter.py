import re
from typing import List, Tuple, Any
import logging

import cv2 as cv
import numpy as np

from util import Util

class FrameProcessor:
    SWING_CLIP_PREFIX: str = "video/swings/"
    NEED_TRANSPOSE: bool = False
    NEED_FLIP: bool = False
    INPUT_SCALE: float = 0.7

    def __init__(self, filename=None, win_name=None) -> None:
        self.filename: str = filename
        self.processor_name: str = __file__
        self.win_name: str = win_name
        self.start_zone: StartZone = StartZone(win_name, need_load=False)

    def process_frame(self, frame: np.ndarray, frame_cnt: int, zone_draw_mode: bool = False) -> np.ndarray:
        # if frame_cnt == 1:
        #     pass  # OnePointZone.reset_zone_point()  # it points to ball so we have to re-init it each time

        if FrameProcessor.INPUT_SCALE != 1.0:
            cv.resize(frame, None, fx=FrameProcessor.INPUT_SCALE, fy=FrameProcessor.INPUT_SCALE)  # !!!
        if FrameProcessor.NEED_TRANSPOSE:
            frame = cv.transpose(frame)
        if FrameProcessor.NEED_FLIP:
            frame = cv.flip(frame, 1)

        if self.start_zone.click_xy is None: # todo change
            return frame
        if self.start_zone.zone_x is None: # todo change
            if not self.start_zone.find_zone(frame):
                print(" Error!!! ball was clicked however Start Zone cannot be found!")
                return frame

        start_zone_state = self.start_zone.get_current_state(frame)
        History.save_state(start_zone_state, frame)

        r = re.search('B{7}B*[MB]{0,7}E{15}$', History.states_string)  # B{7}[MB]*E{7}$
        if r:
            History.write_swing(r)
            History.reset()

        if zone_draw_mode:
            frame = self.start_zone.draw(frame)
        return frame

    def __del__(self):
        self.start_zone.save()


# class StartBall:
#     x, y = None, None
#     contour = None
#     area: float = None
#
#     def __init__(self,x,y,contour):
#         self.x, self.y = x, y
#         self.contour = contour
#         self.area = cv.contourArea(contour)
#         logging.debug(f" StartBall is set: {x=} {y=} {len(self.contour)=} {self.area=}")


class StartZone:
    BLUR_LEVEL: int = 3
    MAX_BALL_SIZE: int = int(30 * FrameProcessor.INPUT_SCALE)  # how far from click_xy we should search for ball contour
    click_xy: Tuple[int, int] = None
    ball_contour: np.ndarray = None
    ball_area: float = None
    # corner_lst: List[Tuple[int, int]] = None
    # zone_contour: List[Any] = None
    thresh_val: float = None
    zone_x, zone_y, zone_w, zone_h = None, None, None, None  # int
    ball_roi, ball_roi_x, ball_roi_y, ball_roi_w, ball_roi_h = None, None, None, None, None
    zone_state = None

    def __init__(self, win_name, need_load=False):
        self.win_name = win_name
        if need_load:
            self.load()
        cv.setMouseCallback(win_name, self.mouse_callback)

    def find_zone(self, frame) -> bool:
        # try to set up Start Zone (ball, border).
        # return True if ok (found and set up), else - False
        if not self.click_xy:  # ball was not clicked yet
            return False

        # set ball_roi: ROI for potential ball area
        click_x, click_y = self.click_xy
        x_max, y_max = frame.shape[1], frame.shape[0]
        self.ball_roi_x, self.ball_roi_y = max(click_x - self.MAX_BALL_SIZE, 0), max(click_y - self.MAX_BALL_SIZE, 0)
        self.ball_roi_w, self.ball_roi_h = min(x_max - self.ball_roi_x, self.MAX_BALL_SIZE * 2), min(y_max - self.ball_roi_y, self.MAX_BALL_SIZE * 2)
        self.ball_roi = frame[self.ball_roi_y:self.ball_roi_y + self.ball_roi_h, self.ball_roi_x:self.ball_roi_x + self.ball_roi_w]

        # prepare image of ball_roi: bgr->gray->blur->open->close
        gray = cv.cvtColor(self.ball_roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (self.BLUR_LEVEL, self.BLUR_LEVEL), 0)
        kernel = np.ones((self.BLUR_LEVEL, self.BLUR_LEVEL), np.uint8)
        gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
        # Util.show_img(gray, "StartArea: gray", 1)

        # StartArea.thresh_val, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thresh = self.get_best_threshold(gray)
        if not thresh:  # can't found ball contour
            return False
        self.thresh_val, thresh_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
        Util.show_img(thresh_img, "StartArea: thresh_img", 1)

        contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cont_lst = [cont for cont in contours
                    if cv.pointPolygonTest(cont, (click_x - self.ball_roi_x, click_y - self.ball_roi_y), measureDist=False) >= 0]
        if len(cont_lst) == 1:
            x, y, w, h = cv.boundingRect(cont_lst[0])
            d = max(w, h)
            if d >= 2 * StartZone.MAX_BALL_SIZE - 4:  # nothing worth is found, contour is for total ball_roi image
                return False

            # found ball contour: include click_xy, not as big as total ball_roi image
            self.ball_contour = cont_lst[0]
            self.ball_area = cv.contourArea(self.ball_contour)
            # self.zone_x, self.zone_y = ball_roi_x + x - 2 * d, ball_roi_y + y - 2 * d  # 2 cells aside
            # self.zone_w, self.zone_h = 5 * d, 5 * d  # 2 cells + original cell + 2 cells
            self.zone_x, self.zone_y = self.ball_roi_x + x - 2 * self.MAX_BALL_SIZE, self.ball_roi_y + y - 2 * self.MAX_BALL_SIZE  # 2 cells aside
            self.zone_w, self.zone_h = 5 * self.MAX_BALL_SIZE, 5 * self.MAX_BALL_SIZE  # 2 cells + original cell + 2 cells
            logging.debug(f"StartArea is set by ball position: {self.zone_x=},{self.zone_y=}   {self.zone_w=},{self.zone_h=}  {self.thresh_val=}")
            return True
        elif len(cont_lst) == 0:  # not found any contour around click_xy
            return False
        elif len(cont_lst) > 1:
            logging.error(
                f"!!!! Error !!! several contours include one point. {self.MAX_BALL_SIZE=}, {cont_lst}=")
        return False

    def zone_is_found(self):
        return False if self.zone_x is None else True

    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            StartZone.click_xy = (x, y)
            StartZone.zone_x = None
        if event == cv.EVENT_RBUTTONDOWN:
            StartZone.click_xy = None
            StartZone.zone_x = None

    def get_best_threshold(self, gray):
        # going through threshold levels to find one which include click_xy and has got a max contour area
        level_results = []
        for thresh in range(50, 200, 5):
            _, img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
            contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # if len(contours) != 1:  # no contours which include click_xy or too much
            #     continue
            cont_lst = [cont for cont in contours if
                        cv.pointPolygonTest(cont, (self.click_xy[0]-self.ball_roi_x,self.click_xy[1]-self.ball_roi_y), measureDist=False) >= 0]
            if len(cont_lst) != 1:  # no contours which include click_xy
                continue
            cont = cont_lst[0]  # it should be the one only contour which include click_xy
            area = cv.contourArea(cont)
            x, y, w, h = cv.boundingRect(cont)
            if max(w, h) >= max(gray.shape) - 2:  # contour for total image is useless
                # logging.debug(f"find_best_thresh: too big contour {w=} {h=} {gray.shape=}")
                continue
            # Util.show_img(img, "thresh find", 0)
            level_results.append({"thresh": thresh, "area": area, "d": max(w, h)})
        level_results = sorted(level_results, key=lambda result: result["area"], reverse=True)
        thresh_otsu, _ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # for debug check only
        logging.debug(f"{len(level_results)=} {thresh_otsu=}")
        if len(level_results) > 1:  # return second best by area if possible
            return level_results[1]["thresh"]
        if len(level_results) == 1:  # return just best if there is only one result
            return level_results[0]["thresh"]
        return None

    def get_current_state(self, frame) -> str:
        # analyze current state of StartArea: 'E' - empty, 'B' - ball, 'M' - mess
        roi = frame[self.zone_y:self.zone_y + self.zone_h, self.zone_x:self.zone_x + self.zone_w]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (self.BLUR_LEVEL, self.BLUR_LEVEL), 0)
        kernel = np.ones((self.BLUR_LEVEL, self.BLUR_LEVEL), np.uint8)
        gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
        # Util.show_img(gray, "gray", 1)

        _, thresh_img = cv.threshold(gray, self.thresh_val, 255, cv.THRESH_BINARY)
        Util.show_img(thresh_img, "Stream:   thresh_img", 1)

        contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if self.ball_area * 0.5 < cv.contourArea(cnt)]
        all_cnt = len(contours)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) < self.ball_area * 2]
        logging.debug(f"get_status: found contours: all = {all_cnt} not_big = {len(contours)} ")

        if all_cnt == 0:
            self.zone_state = 'E'
            return self.zone_state
        if len(contours) == 1 and not self.is_touched_border(contours[0]):
            match_rate = cv.matchShapes(contours[0], self.ball_contour, 1, 0)
            logging.debug(f" {match_rate=}")
            if match_rate < 0.5:
                self.zone_state = 'B'
                return self.zone_state
        self.zone_state = 'M'
        return self.zone_state

    def is_touched_border(self,contour):
        x, y, w, h = cv.boundingRect(contour)
        return True \
            if x == self.zone_x or y == self.zone_y or x + w == self.zone_x + self.zone_w or y + h == self.zone_y + self.zone_h \
            else False

    def draw(self, frame):
        cv.rectangle(frame, (self.zone_x, self.zone_y), (self.zone_x + self.zone_w, self.zone_y + self.zone_h), (255, 0, 0), 1)
        cv.drawMarker(frame, self.click_xy, (0, 0, 255), cv.MARKER_CROSS, 20, 1)
        # # cv.drawContours(frame, [StartArea.contour], 0, (0, 0, 255), 3)
        cv.putText(frame, f"{self.zone_state}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        return frame

    def load(self):
        pass

    def save(self):
        pass


class History:
    states_string: str = ""

    @classmethod
    def save_state(cls, state: str, frame):
        # status_history += status
        # frames_buffer.append(frame.copy())
        # logging.debug(f"{len(frames_buffer)=}")
        pass

    @classmethod
    def write_swing(cls, r):
        pass

    @classmethod
    def reset(cls):
        # status_history = ''
        # frames_buffer.clear()
        pass
