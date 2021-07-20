import cv2 as cv
import numpy as np

import logging

logging.basicConfig(filename='debug.log', level=logging.DEBUG)


class ZonePoints:
    image = None
    zone_corners_lst = []

    @classmethod
    def set_mouse(cls, win_name):
        logging.debug("logging set on")
        # ZonePoints.reset_zone_corner_lst()  # temporally until save/load
        cv.setMouseCallback(win_name, mouse_callback)

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
        logging.debug("logging draw corners")
        print(cls.zone_corners_lst)
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
    zone_draw_mode = False  # True - draw active zone (corners_lst) on all images
    # img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')
    ZonePoints.set_mouse('image')

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

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
