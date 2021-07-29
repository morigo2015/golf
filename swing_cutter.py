import re
from typing import List, Tuple, Any
from collections import deque
import logging
import datetime

import cv2 as cv
import numpy as np

from util import Util, FrameStream, WriteStream

logging.basicConfig(filename='debug.log', level=logging.DEBUG)


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

        if not self.start_zone.ball_is_clicked():
            return frame
        if not self.start_zone.zone_is_found():
            if not self.start_zone.get_zone(frame):
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

class ROI:
    x, y, w, h = [None] * 4

    def __init__(self, frame_shape, point: Tuple[int, int], roi_size: int):
        self.w, self.h = [roi_size] * 2
        self.x, self.y = point[0] - int(self.w / 2), point[1] - int(self.h / 2)
        self.trim_at_bounds(frame_shape)
        # logging.debug(f"init: {self=}")

    def trim_at_bounds(self, frame_shape):
        x_max, y_max = frame_shape[1], frame_shape[0]
        self.x, self.y = max(self.x, 0), max(self.y, 0)
        self.x, self.y = min(self.x, x_max), min(self.y, y_max),
        self.w, self.h = min(self.w, x_max - self.x), min(self.h, y_max - self.y),
        self.w, self.h = max(self.w, 0), max(self.h, 0)

    def extract_roi(self, frame):
        return frame[self.y: self.y + self.h, self.x: self.x + self.w]

    def is_touched_border(self, contour):
        x, y, w, h = cv.boundingRect(contour)
        return True if x == self.x or y == self.y or x + w == self.x + self.w or y + h == self.y + self.h else False

    def draw(self, frame):
        cv.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 1)
        return frame

    def __repr__(self):
        return f"roi({self.x=},{self.y=},{self.w=},{self.h=})"


class StartZone:
    BLUR_LEVEL: int = 3
    MAX_BALL_SIZE: int = int(25 * FrameProcessor.INPUT_SCALE)  # how far from click_xy we should search for ball contour
    ZONE_BALL_RATIO: int = 5
    click_xy: Tuple[int, int] = None
    ball_contour: np.ndarray = None
    ball_area: float = None
    # corner_lst: List[Tuple[int, int]] = None
    # zone_contour: List[Any] = None
    thresh_val: float = None
    ball_roi: ROI = None
    ball_roi_img = None
    zone_roi: ROI = None
    zone_state = None
    closest_brightest = None

    @staticmethod
    def zone_reset():
        StartZone.click_xy, StartZone.ball_contour, StartZone.ball_area, StartZone.thresh_val, StartZone.ball_roi, StartZone.ball_roi_img = [None] * 6
        StartZone.closest_brightest, StartZone.zone_roi, StartZone.zone_state = [None] * 3

    def __init__(self, win_name, need_load=False):
        self.win_name = win_name
        if need_load:
            self.load()
        cv.setMouseCallback(win_name, self.mouse_callback)

    def preprocess_image(self, roi_img):
        # prepare image of ball_roi: bgr->gray->blur->open->close
        gray = cv.cvtColor(roi_img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (self.BLUR_LEVEL, self.BLUR_LEVEL), 0)
        kernel = np.ones((self.BLUR_LEVEL, self.BLUR_LEVEL), np.uint8)
        gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
        Util.show_img(gray, "StartArea: gray", 1)
        return gray

    def get_zone(self, frame) -> bool:
        # try to set up Start Zone (ball, border).
        # return True if ok (found and set up), else - False
        if not self.click_xy:  # ball was not clicked yet
            return False

        self.ball_roi = ROI(frame.shape, self.click_xy, self.MAX_BALL_SIZE * 5)
        self.ball_roi_img = self.ball_roi.extract_roi(frame)

        # prepare image of ball_roi: bgr->gray->blur->open->close
        # gray = cv.cvtColor(self.ball_roi_img, cv.COLOR_BGR2GRAY)
        # gray = cv.GaussianBlur(gray, (self.BLUR_LEVEL, self.BLUR_LEVEL), 0)
        # kernel = np.ones((self.BLUR_LEVEL, self.BLUR_LEVEL), np.uint8)
        # gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        # gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)

        gray = self.preprocess_image(self.ball_roi_img)
        # Util.show_img(gray, "StartArea: gray", 1)
        # brightest_xy = self.get_closest_brightest(gray)  # find brightest point near click_xy

        # StartArea.thresh_val, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thresh = self.get_best_threshold(gray)  # , brightest_xy)
        if not thresh:  # can't found ball contour
            return False
        self.thresh_val, thresh_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
        Util.show_img(thresh_img, "StartArea: thresh_img", 1)

        contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cont_lst = contours
        # cont_lst = [cont for cont in contours
        #             if cv.pointPolygonTest(cont, brightest_xy, measureDist=False) >= 0]
        if len(cont_lst) == 1:
            x, y, w, h = cv.boundingRect(cont_lst[0])
            d = max(w, h)
            if d >= 2 * StartZone.MAX_BALL_SIZE - 4:  # nothing worth is found, contour is for total ball_roi image
                return False

            # found ball contour: include click_xy, not as big as total ball_roi image
            self.ball_contour = cont_lst[0]
            self.ball_area = cv.contourArea(self.ball_contour)
            zone_center_point = self.xy_zone_2_frame((int(x + d / 2), int(y + d / 2)))
            self.zone_roi = ROI(frame.shape, zone_center_point, self.MAX_BALL_SIZE * self.ZONE_BALL_RATIO)
            logging.debug(f"StartArea is set by ball position: {self.zone_roi=}  {self.thresh_val=}")
            return True
        elif len(cont_lst) == 0:  # not found any contour around click_xy
            return False
        elif len(cont_lst) > 1:
            logging.error(
                f"!!!! Error !!! several contours include one point. {self.MAX_BALL_SIZE=}, {cont_lst}=")
        return False

    @staticmethod
    def get_best_threshold(gray):  # , point_xy):
        # going through threshold levels to find one which include point_xy and has got a max contour area
        level_results = []
        for thresh in range(50, 200, 5):
            _, img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
            contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            all_cont_cnt = len(contours)
            if len(contours) != 1:
                continue
            # Util.show_img(img, f"thresh level = {thresh}", 1)
            cont_lst = contours  # just stub
            # cont_lst = [cont for cont in contours if
            #             cv.pointPolygonTest(cont, point_xy, measureDist=False) >= 0]
            # if len(cont_lst) != 1:  # no contours which include click_xy
            #     logging.debug(f"get_best_thresh: len!=1 {thresh=} {all_cont_cnt=} {len(cont_lst)=} ")
            #     continue
            cont = cont_lst[0]  # it should be the one only contour which include click_xy
            area = cv.contourArea(cont)
            x, y, w, h = cv.boundingRect(cont)
            if max(w, h) >= max(gray.shape) - 6:  # contour for total image is useless
                # logging.debug(f"find_best_thresh: too big contour {w=} {h=} {gray.shape=}")
                logging.debug(f"get_best_thresh: too big {thresh=} {all_cont_cnt=} {len(cont_lst)=} {w=} {h=} ")
                continue
            result = {"thresh": thresh, "area": area, "d": max(w, h)}
            level_results.append(result)
            logging.debug(f"get_best_thresh: add result {result=}  ")
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

        # roi = frame[self.zone_y:self.zone_y + self.zone_h, self.zone_x:self.zone_x + self.zone_w]
        roi_img = self.zone_roi.extract_roi(frame)
        # gray = cv.cvtColor(roi_img, cv.COLOR_BGR2GRAY)
        # gray = cv.GaussianBlur(gray, (self.BLUR_LEVEL, self.BLUR_LEVEL), 0)
        # kernel = np.ones((self.BLUR_LEVEL, self.BLUR_LEVEL), np.uint8)
        # gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        # gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
        gray = self.preprocess_image(roi_img)
        Util.show_img(gray, "Stream:   gray", 1)

        _, thresh_img = cv.threshold(gray, self.thresh_val, 255, cv.THRESH_BINARY)
        Util.show_img(thresh_img, "Stream:   thresh_img", 1)

        contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if self.ball_area * 0.5 < cv.contourArea(cnt)]
        all_cnt = len(contours)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) < self.ball_area * 2]
        # logging.debug(f"get_status: found contours: all = {all_cnt} not_big = {len(contours)} ")

        if all_cnt == 0:
            self.zone_state = 'E'
            return self.zone_state
        if len(contours) == 1 and not self.zone_roi.is_touched_border(contours[0]):
            match_rate = cv.matchShapes(contours[0], self.ball_contour, 1, 0)
            # logging.debug(f" {match_rate=}")
            if match_rate < 0.5:
                self.zone_state = 'B'
                return self.zone_state
        self.zone_state = 'M'
        return self.zone_state

    def zone_is_found(self):
        return False if self.zone_roi is None else True

    def ball_is_clicked(self):
        return False if self.click_xy is None else True

    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            StartZone.zone_reset()
            StartZone.click_xy = (x, y)
        if event == cv.EVENT_RBUTTONDOWN:
            StartZone.zone_reset()

    def draw(self, frame):
        cv.rectangle(frame,
                     (self.zone_roi.x, self.zone_roi.y), (self.zone_roi.x + self.zone_roi.w, self.zone_roi.y + self.zone_roi.h), (255, 0, 0), 1)
        cv.drawMarker(frame, self.click_xy, (0, 0, 255), cv.MARKER_CROSS, 20, 1)
        # cv.drawMarker(frame, self.xy_zone_2_frame(self.closest_brightest), (0, 255, 0), cv.MARKER_CROSS, 20, 1)
        # # cv.drawContours(frame, [StartArea.contour], 0, (0, 0, 255), 3)
        cv.putText(frame, f"{self.zone_state}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        return frame

    def load(self):
        pass

    def save(self):
        pass

    def xy_zone_2_frame(self, zone_xy):
        return zone_xy[0] + self.ball_roi.x, zone_xy[1] + self.ball_roi.y

    def xy_frame_2_zone(self, frame_xy):
        return frame_xy[0] - self.ball_roi.x, frame_xy[1] - self.ball_roi.y

    # def get_closest_brightest(self, gray):
    #     # return coordinate of closest to click_xy point which is == to max(gray)
    #     click_x, click_y = self.xy_frame_2_zone(self.click_xy)
    #     brightest_points = np.where(gray == np.amax(gray))  # tuple of 2 arrays: x[] and y[] of grightest points
    #     brightest_points_lst = list(zip(brightest_points[0], brightest_points[1]))  # list of tuples (x,y) of brightest points
    #     closest_point = sorted(brightest_points_lst, key=lambda xy: np.sqrt((xy[0] - click_x) ** 2 + (xy[1] - click_y) ** 2))[0]
    #     logging.debug(f"get_closest_brightest for {self.click_xy=} ({click_x=},{click_y=}) -->  {closest_point=}")
    #     self.closest_brightest = int(closest_point[0]), int(closest_point[1])
    #     return self.closest_brightest


class History:
    FRAME_BUFF_SZ = 300
    MAX_CLIP_SZ = 150
    frames_buffer = deque(maxlen=FRAME_BUFF_SZ)
    states_string: str = ""

    @classmethod
    def save_state(cls, state: str, frame):
        cls.states_string += state
        cls.frames_buffer.append(frame.copy())
        # logging.debug(f"{len(cls.frames_buffer)=}")

    @classmethod
    def write_swing(cls, r):
        start_pos, end_pos = r.span()
        frames_to_write = min(end_pos - start_pos, cls.MAX_CLIP_SZ)
        frames_to_skip = len(cls.frames_buffer) - frames_to_write
        for i in range(frames_to_skip):
            cls.frames_buffer.popleft()
        out_file_name = f"{FrameProcessor.SWING_CLIP_PREFIX}{datetime.datetime.now().strftime('%H:%M:%S')}.avi"  # f"{swing_clip_prefix}{swing_clip_cnt}.avi"
        out_fs = WriteStream(out_file_name, fps=5)
        for i in range(frames_to_write):
            out_frame = cls.frames_buffer.popleft()
            out_fs.write(out_frame)
        del out_fs
        logging.debug(f"swing clip written: {out_file_name=} {start_pos=} {end_pos=}")
        return out_file_name

    @classmethod
    def reset(cls):
        cls.status_history = ''
        cls.frames_buffer.clear()


def test_roi():
    input_fs = FrameStream("video/out2.avi")
    roi_sz = 300
    frame, _, _ = input_fs.next_frame()
    cv.namedWindow('tst_roi')
    start_zone = StartZone('tst_roi')
    old_click_xy = (-1, -1)
    while cv.waitKey(1) != ord('q'):
        cv.imshow('tst_roi', frame)
        if start_zone.click_xy is not None:
            roi = ROI(frame.shape, start_zone.click_xy, roi_sz)
            roi_img = roi.extract_roi(frame)
            Util.show_img(roi_img, "roi_img", 1)
            if start_zone.click_xy != old_click_xy:
                logging.debug(f"{roi=}: {frame.shape=} {start_zone.click_xy=} {roi_sz=}")
            frame = roi.draw(frame)
            old_click_xy = start_zone.click_xy


def main():
    test_roi()
    pass


if __name__ == "__main__":
    main()
