# one-point zone (ball position)

import json
import logging

import cv2 as cv

# import numpy as np

param_fname = "param.json"


class OnePointZone:
    image = None
    zone_point = None

    @classmethod
    def zone_init(cls, win_name, need_load=False):
        logging.debug("logging init")
        if need_load:
            try:
                with open(param_fname, 'r') as f:
                    cls.zone_point = json.load(f)
                    logging.debug(f"load zone_corner_lst from {param_fname}: {cls.zone_point} ")
            except FileNotFoundError as error:
                logging.debug("param.json not found. reset zone_corner_lst")
                cls.reset_zone_point()
        else:
            cls.reset_zone_point()
        cv.setMouseCallback(win_name, cls.mouse_callback)

    @classmethod
    def zone_save(cls):
        with open(param_fname, 'w') as f:
            json.dump(cls.zone_point, f, indent=2)
            logging.debug(f"save zone_corner_lst to {param_fname}: {cls.zone_point} ")

    @classmethod
    def new_frame(cls, image):
        cls.image = image

    @classmethod
    def draw_zone(cls, image):
        if not cls.zone_point:
            return
        logging.debug(f"draw zone point: {cls.zone_point}")
        # cv.drawContours(image, [cls.__get_zone_contour()], -1, (255, 255, 255), 3)
        cv.drawMarker(image, cls.zone_point, (0, 0, 255), cv.MARKER_CROSS, 20, 1)
        return

    @classmethod
    def zone_is_defined(cls):
        return False if cls.zone_point is None else True

    @classmethod
    def __left_clicked(cls, x, y):
        logging.debug(f"left_clicked {x} {y}")
        # cv.circle(cls.image, (x, y), 5, (0, 0, 255), -1)
        cls.zone_point = (x, y)
        # cls.draw_zone(cls.image)

    @classmethod
    def __right_clicked(cls, x, y):
        cls.reset_zone_point()

    @classmethod
    def reset_zone_point(cls):
        logging.debug("reset zone_corner_lst")
        cls.zone_point = None

    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            OnePointZone.__left_clicked(x, y)
        if event == cv.EVENT_RBUTTONDOWN:
            OnePointZone.__right_clicked(x, y)


# -----------------------------------------------------------------------

from util import FrameStream

# fs = FrameStream("video/clp/bf-merged.avi")
fs = FrameStream("video/raw-done/out2-back-further1.avi")


def main():
    frame_mode = False  # initial frame_mode
    zone_draw_mode = True  # True - draw active zone (corners_lst) on all images
    # img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')
    OnePointZone.zone_init('image')

    frame, _, _ = fs.next_frame()

    while True:
        frame, _, _ = fs.next_frame()
        logging.debug("new frame")

        if frame is None:
            break

        OnePointZone.new_frame(frame)

        # if zone_draw_mode:
        #     OnePointZone.draw_zone(frame)

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

    OnePointZone.zone_save()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
