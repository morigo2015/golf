import math
import time
import os
import glob
import numpy as np
import cv2 as cv

from util import Util, FrameStream

markup_errors = "img/err"
debug = True
# debug = False


class StartArea:
    start_area_bg = None
    corners = None

    def __init__(self, start_point, tl_dist, img):
        # start_area = (left,right,up,down) extended to left from starting point
        half_side = int(tl_dist * 0.7 / 2)
        self.corners = (start_point[0] - half_side, start_point[1] - half_side), \
                       (start_point[0] + half_side, start_point[1] + half_side)
        area = self._get_area(img)
        self.start_area_bg = self._get_start_area_bg(area)
        self.ball_center, self.ball_axis, self.ball_angle = None, None, None
        self.ball_stat = []

    def _get_area(self, img):
        l, r, u, b = self.corners[0][0], self.corners[1][0], self.corners[0][1], self.corners[1][1]
        area = img[u:b, l:r].copy()
        return area

    def _get_start_area_bg(self, img):
        img = cv.medianBlur(img, 5)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.medianBlur(gray, 5)
        return gray

    def draw_start_area(self, image, color=(255, 100, 100)):
        cv.rectangle(image, self.corners[0], self.corners[1], color, 1)

    def got_ball(self, image):
        ball_approx_area = 1600  # 2000 # примерно площадь мяча (на среднем расстоянии)

        start_area = self._get_area(image)

        img = cv.medianBlur(start_area, 5)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.medianBlur(gray, 5)
        diff = cv.absdiff(self.start_area_bg, gray)
        thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)[1]
        thresh = cv.dilate(thresh, None, iterations=2)
        cv.imshow("thresh", thresh)
        changed_area_share = thresh.mean() / 255.
        ball_to_area_share = ball_approx_area / thresh.size
        if not (ball_to_area_share * 0.5 < changed_area_share < ball_to_area_share * 2):
            return False

        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ball_cont_lst = [c for c in contours if ball_approx_area * 0.5 < cv.contourArea(c) < ball_approx_area * 2]
        if len(ball_cont_lst) != 1:
            return False
        cont = ball_cont_lst[0]

        hull = cv.convexHull(cont)
        convex_ratio = cv.contourArea(cont) / cv.contourArea(hull)
        if not (convex_ratio > 0.95):
            # print(f"not convex contour. ratio={convex_ratio}")
            return False  # non-convex contour

        ellipse = cv.fitEllipse(cont)
        print(f"center=({ellipse[0][0]:.0f},{ellipse[0][1]:.0f}) "
              f"ratio={max(ellipse[1]) / min(ellipse[1])} {cv.contourArea(cont)=} {thresh.mean()/255.=}")
        self.ball_center = self.corners[0][0] + ellipse[0][0], self.corners[0][1] + ellipse[0][1]
        self.ball_axis = ellipse[1]
        self.ball_angle = ellipse[2]
        self.ball_stat += [{'thresh_mean': thresh.mean() / 255., 'area': cv.contourArea(cont),
                            'area_to_size': cv.contourArea(cont) / thresh.size,
                            'axes_ratio': max(ellipse[1]) / min(ellipse[1])}]
        return True

    def draw_ball(self, img, color=(255, 100, 100)):
        cv.ellipse(img, (self.ball_center, self.ball_axis, self.ball_angle), color)

    def print_ball_stat(self):
        print("ball search stats:")
        for k in self.ball_stat[0]:
            vals = [val[k] for val in self.ball_stat]
            print(f"{k}:  {min(vals):.2f} - {max(vals):.2f}")
