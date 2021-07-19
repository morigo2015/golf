import cv2 as cv
import numpy as np


class ZonePoints:
    image = None
    zone_corners_lst = None

    @classmethod
    def set_zone_on(cls, win_name, image):
        cls.image = image
        cls.zone_corners_lst = []
        cv.setMouseCallback(win_name, cls.mouse_callback)

    @classmethod
    def set_zone_off(cls):
        pass

    @classmethod
    def draw_zone_corners(cls):
        print(cls.zone_corners_lst)
        contour = np.array(cls.zone_corners_lst).reshape((-1, 1, 2)).astype(np.int32)
        cv.drawContours(cls.image, [contour], -1, (255, 255, 255), cv.FILLED)

    @classmethod
    def add_zone_corner(cls, x, y):
        cv.circle(cls.image, (x, y), 5, (0, 0, 255), -1)
        cls.zone_corners_lst.append((x, y))
        cls.draw_zone_corners()

    @classmethod
    def reset_zone_corner_lst(cls):
        pass

    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            ZonePoints.add_zone_corner(x, y)


# -----------------------------------------------------------------------

from util import FrameStream

fs = FrameStream("video/clp/bf-merged.avi")


def main():
    frame_mode = False  # initial frame_mode
    zone_mode = False
    # img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')

    img, _, _ = fs.next_frame()
    ZonePoints.set_zone_on('image', img)

    while (1):
        img, _, _ = fs.next_frame()

        if img is None:
            break

        cv.imshow('image', img)
        ch = cv.waitKey(0 if frame_mode else 1)
        if ch == ord('z'):
            zone_mode = not zone_mode
        elif ch == ord(' '):
            frame_mode = True
            ZonePoints.set_zone_on('image', img)
            continue
        elif ch == ord('g'):
            frame_mode = False
            ZonePoints.set_zone_off()
            continue
        elif ch == 27 or ch == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
