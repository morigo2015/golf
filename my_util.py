import logging
import math
import glob
import time
import numpy as np
import cv2 as cv
from typing import List, Tuple, Union, Any, TypeVar, Dict, Deque

# type hints abbreviations since current version of Python doesn't support |None in hints
POINT = Tuple[int, int]
POINT_ = TypeVar('POINT_', POINT, type(None))
NDARRAY = np.ndarray
NDARRAY_ = TypeVar('NDARRAY_', NDARRAY, type(None))
float_ = TypeVar('float_', float, type(None))
str_ = TypeVar('str_', str, type(None))


class Util:
    out_fname = "img/res"

    @staticmethod
    def int2(p):
        return int(p[0]), int(p[1])

    @staticmethod
    def get_angle(p1=None, p2=None, pts=None):
        if pts is not None:
            p1, p2 = pts
        dw = p2[0] - p1[0]
        dh = p2[1] - p1[1]
        ang = np.rad2deg(math.atan2(dh, dw))
        return ang

    @staticmethod
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            print(f"{line1=} {line2=} {xdiff=}{ydiff=}{div=}")
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def middle(p1, p2):
        return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    @staticmethod
    def rect_2points(rect):
        bp0, bp1, bp2, bp3 = cv.boxPoints(rect)
        # dist01 = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        # dist12 = math.sqrt((box[2][0] - box[1][0]) ** 2 + (box[2][1] - box[1][1]) ** 2)
        if Util.dist(bp0, bp1) > Util.dist(bp1, bp2):  # sides 0-1,2-3 are long, 1-2,3-0 are short
            p1, p2 = Util.middle(bp1, bp2), Util.middle(bp3, bp0)
        else:  # sides 0-1,2-3 are short, 1-2,3-0 are long
            p1, p2 = Util.middle(bp0, bp1), Util.middle(bp2, bp3)
        return p1, p2

    @staticmethod
    def rect_length(rect):
        return max(rect[1][0], rect[1][1])  # max(w,h)

    @staticmethod
    def rect_area(rect):
        return rect[1][0] * rect[1][1]

    @staticmethod
    def join_rect_lst(lst1, lst2):
        # join 2 lists of rotated rectangles descriptors (idx, rect)*
        # for intersected rects leave one which is longer
        to_remove_1 = []  # list of indexes to remove from lst1
        to_remove_2 = []  # list of indexes to remove from lst2
        for i1, rect1 in enumerate(lst1):
            for i2, rect2 in enumerate(lst2):
                is_intersect, intersect_contour = cv.rotatedRectangleIntersection(rect1, rect2)
                if is_intersect:
                    intersect_area = cv.contourArea(intersect_contour)
                    if intersect_area / min(Util.rect_area(rect1), Util.rect_area(rect2)) > 0.9:
                        if Util.rect_length(rect2) > Util.rect_length(rect1):
                            to_remove_1.append(i1)
                        else:
                            to_remove_2.append(i2)
        # if debug:
        #     print(f"join rect list: remove {len(to_remove_1)} from sticks and {len(to_remove_2)} from rect_lst")
        # joined_lst = np.delete(lst1, to_remove_1).tolist() + np.delete(lst2, to_remove_2).tolist()
        joined_lst = [rect for i, rect in enumerate(lst1) if i not in to_remove_1] + \
                     [rect for i, rect in enumerate(lst2) if i not in to_remove_2]
        return joined_lst

    @staticmethod
    def show_img(img, name="",delay=0):
        if delay <0: # just skip this debug
            return
        cv.imshow(f"{name}", img)
        ch = cv.waitKey(delay)
        if ch == ord('s'):
            cv.imwrite(f"{Util.out_fname}{name}.png", img)


class FrameStream:
    def __init__(self, source_path):
        self.path = source_path
        self.frame_cnt = 0

        if source_path[-4:] == ".png":
            self.mode = 'images'
            self.file_name_iter = iter(sorted(glob.glob(source_path)))
        else:
            self.mode = 'video'
            self.cap = cv.VideoCapture(source_path)
            if not self.cap.isOpened():
                print(f"Cannot open source {source_path}")
                exit()
        self.start = time.time()

    def next_frame(self):
        self.frame_cnt += 1
        if self.mode == 'images':
            try:
                file_name = next(self.file_name_iter)
                frame = cv.imread(file_name)
                frame_name = file_name
            except StopIteration:
                frame = None
                frame_name = "End_of_list"
        else:
            ret, frame = self.cap.read()
            frame_name = f"Frame_{self.frame_cnt}"
            if not ret:
                frame = None
        return frame, frame_name, self.frame_cnt

    def total_time(self):
        return time.time()-self.start

    def fps(self):
        return self.frame_cnt / self.total_time()

    def __del__(self):
        if self.mode == 'video':
            self.cap.release()

class WriteStream:
    # запись в файл, отложенное открытие потока (до первого кадра для определения shape)

    def __init__(self, file_name, fps=20.0, fourcc='XVID'):
        self.file_name = file_name
        self.fourcc = cv.VideoWriter_fourcc(*fourcc)
        self.fps = fps
        self.out = None

    def write(self, out_frame):
        if not self.out:
            out_shape = (out_frame.shape[1], out_frame.shape[0])
            self.out = cv.VideoWriter(self.file_name, self.fourcc, self.fps, out_shape)
        self.out.write(out_frame)

    def write_bw(self, bw_img, text=None):
        # write black_white (one channel) image with colour text added
        colour_img = cv.cvtColor(bw_img, cv.COLOR_GRAY2BGR)
        colour_img = cv.resize(colour_img, (500, 500))
        cv.putText(colour_img, text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.write(colour_img)

    def __del__(self):
        pass
        # self.out.release() # todo вернуть обратно
        # logging.debug(f"file {self.file_name} released")

# -------------------------------------------------------------------------------------------------------------

def main():
    print(f"{Util.dist((100, 100), (200, 200))}")


if __name__ == '__main__':
    main()
