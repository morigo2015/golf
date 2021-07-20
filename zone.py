import json
import logging

import cv2 as cv
import numpy as np

param_fname = "param.json"

logging.basicConfig(filename='debug.log', level=logging.DEBUG)


class ZonePoints:
    image = None
    zone_corners_lst = []

    @classmethod
    def zone_init(cls, win_name):
        logging.debug("logging init")
        try:
            with open(param_fname, 'r') as f:
                cls.zone_corners_lst = json.load(f)
                logging.debug(f"load zone_corner_lst from {param_fname}: {cls.zone_corners_lst} ")
        except FileNotFoundError as error:
            logging.debug("param.json not found. reset zone_corner_lst")
            cls.reset_zone_corner_lst()
        # ZonePoints.reset_zone_corner_lst()  # temporally until save/load
        cv.setMouseCallback(win_name, mouse_callback)

    @classmethod
    def zone_save(cls):
        with open(param_fname, 'w') as f:
            json.dump(cls.zone_corners_lst, f, indent=2)
            logging.debug(f"save zone_corner_lst to {param_fname}: {cls.zone_corners_lst} ")

    @classmethod
    def new_frame(cls, image):
        cls.image = image

    @classmethod
    def add_zone_corner(cls, x, y):
        logging.debug(f"logging add_zone_corner {x} {y}")
        cv.circle(cls.image, (x, y), 5, (0, 0, 255), -1)
        cls.zone_corners_lst.append((x, y))
        # cls.draw_zone_corners(image)

    @classmethod
    def reset_zone_corner_lst(cls):
        logging.debug("logging reset zone_corner_lst")
        cls.zone_corners_lst = []

    @classmethod
    def draw_zone_corners(cls, image):
        logging.debug(f"logging draw corners: {cls.zone_corners_lst}")
        if not len(cls.zone_corners_lst):
            return
        contour = np.array(cls.zone_corners_lst).reshape((-1, 1, 2)).astype(np.int32)
        cv.drawContours(image, [contour], -1, (255, 255, 255), 3)
        return


def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        ZonePoints.add_zone_corner(x, y)
    if event == cv.EVENT_RBUTTONDOWN:
        ZonePoints.reset_zone_corner_lst()


# -----------------------------------------------------------------------

from util import FrameStream

# fs = FrameStream("video/clp/bf-merged.avi")
fs = FrameStream("video/raw-done/out2-back-further1.avi")


def main():
    frame_mode = False  # initial frame_mode
    zone_draw_mode = True  # True - draw active zone (corners_lst) on all images
    # img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')
    ZonePoints.zone_init('image')

    frame, _, _ = fs.next_frame()

    while True:
        frame, _, _ = fs.next_frame()
        logging.debug("new frame")

        if frame is None:
            break

        ZonePoints.new_frame(frame)

        if zone_draw_mode:
            ZonePoints.draw_zone_corners(frame)

        cv.imshow('image', frame)
        ch = cv.waitKey(0 if frame_mode else 1)

        if ch == ord('z'):
            zone_draw_mode = not zone_draw_mode
        elif ch == ord(' '):
            frame_mode = True
            continue
        elif ch == ord('g'):
            frame_mode = False
            continue
        elif ch == 27 or ch == ord('q'):
            break

    ZonePoints.zone_save()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
