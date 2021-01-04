import math
import time
import os
import glob
import numpy as np
import cv2 as cv

from util import Util, FrameStream
from start_area import StartArea

markup_errors = "img/err"
# debug = True
debug = False


class MarkUp:
    sticks = None  # list of 3 sticks reverse ordered by distance to golfer (increasing h): [(idx, (center,size,angle))]
    markup = None  # high,low,vert, start_point, aim_line_angle, cross_hv, cross_lv
    start_area = None

    def __init__(self, img, name=""):

        if img is None:
            print(f"Error!! No input image in {name}")
            return
        if img.shape != (1080, 1920, 3):
            print(f"Error!! Illegal image. Shape={img.shape} instead of (1080,1920,3) in {name}")
            return
        # if debug:
        #     Util.show_img(img, "_start_img")

        self.sticks = _get_sticks(img, name)  # get list of black sticks in image
        err_flg, self.markup = _get_markup(self.sticks, img, name)  # find markup pattern (top,low,vert)
        if err_flg:
            print(f"{name}: Markup is not created! Img is stored in {markup_errors}")
            return

        self.start_area = StartArea(self.markup['start_point'], self.markup['tl_dist'], img)

        # if debug:
        #     self.show_markup(img)

    def draw_markup(self, image):
        # draw_markup
        cv.line(image, self.markup['pleft'], self.markup['pright'], (0, 80, 0), 1)  # aiming line
        cv.line(image, self.markup['cross_hv'], self.markup['cross_lv'], (0, 80, 0), 2)
        cv.circle(image, self.markup['start_point'], 3, (0, 80, 0), 1)  # starting cross
        return image

    def show_markup(self, img):
        img = img.copy()
        self.draw_markup(img)
        Util.show_img(img, "draw_markup markup")


def _get_sticks(img, name=""):
    # image processing
    # blur -> convert to gray -> blur -> threshold (3 levels) -> dilate -> get contours ->
    # -> filter by ares (>5000) -> filter by stick shape (length/width>7) -> sort by center_h

    img = img.copy()
    img = cv.medianBlur(img, 5)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray, 5)
    # Util.show_img(gray, "gray")

    sticks = []
    for lim in [3, 7, 10, 15, 25]:  # [3, 5, 8, 15, 25]
        mask = gray < lim
        # mask2 = np.bitwise_and(lim1 <= gray, gray < lim2)
        mask = mask.astype(np.uint8) * 255
        # if debug:
        #     Util.show_img(mask, f"mask for lim={lim}")

        mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)  # 3

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # if debug:
        #     cv.drawContours(img, contours, -1, (0, 255, 0), 3)
        #     cv.imshow(f"lim={lim} contours", img)
        #     cv.waitKey(0)

        rect_lst = [cv.minAreaRect(cnt) for cnt in contours if cv.contourArea(cnt) > 2500]  # 3000
        # if debug:
        # draw_rect(img, rect_lst, color=(255, 255, 0))
        # cv.imshow(f"lim={lim} after area filter", img)
        # cv.waitKey(0)

        rect_lst = [rect for rect in rect_lst if max(rect[1]) / min(rect[1]) > 6]  # rect[1] - size tuple
        # if debug:
        # draw_rect(img, rect_lst, color=(255, 255, 255))
        # cv.imshow(f"lim={lim} after stick shape filter", img)
        # cv.waitKey(0)

        sticks = Util.join_rect_lst(sticks, rect_lst)  # add rect_lst to sticks while removing intersected ones

    sticks = sorted(sticks, key=lambda s: s[0][1])

    # if debug:
    #     print(f"{len(sticks)=} {sticks=}")
    #     draw_rect(img, sticks, "final:")
    #     Util.show_img(img, f"Final for {name}")

    return sticks


def _get_markup(sticks, img=None, name=""):
    # find markup pattern (top,low,vert)
    # go through all triplets (any 3 sticks from the list) checking conditions:
    #   top higher low higher vert                   top.center_y < low.center_y < vert.center_y
    #   top and low are close to parallel
    #   low and vert close to perpendicular
    #   top and low are close enough                 dist(top.center,low.center) < shortest stick
    #   vert and low are close enough                dist(low.center,vert.center) < vert length

    markup = {}
    triplets = []

    for top_ind in range(0, len(sticks)):
        top = sticks[top_ind]
        top_pts = Util.rect_2points(top)
        top_ang = Util.get_angle(pts=top_pts)
        for low_ind in range(top_ind + 1, len(sticks)):
            low = sticks[low_ind]
            low_pts = Util.rect_2points(low)
            low_ang = Util.get_angle(pts=low_pts)
            if debug:
                print(f"{top_ind=} {top[0][1]=:.0f} {low_ind=} {low[0][1]=:.0f}")
            if not (top[0][1] < low[0][1]):  # top over low: top center_y > low center_y
                continue
            if not (abs(top_ang - low_ang) < 13):  # top and low must be close to parallel
                continue
            for vert_ind in range(low_ind + 1, len(sticks)):
                vert = sticks[vert_ind]
                vert_pts = Util.rect_2points(vert)
                vert_ang = Util.get_angle(pts=vert_pts)
                if not (low[0][1] < vert[0][1]):  # low over vert
                    continue
                if not (abs(abs(low_ang - vert_ang) - 90) < 15):  # low and vert must be close to perpendicular
                    continue
                dist_tl = Util.dist(top[0], low[0])
                if not (dist_tl < min(Util.rect_length(top), Util.rect_length(low))):  # dist(top,low) < shortest stick
                    continue
                dist_vl = Util.dist(vert[0], low[0])  # dist from vert center to low center
                if not (dist_vl < Util.rect_length(vert)):  # dist from _centers_ of vert,low < _total_ length of vert
                    continue
                triplets += [(top_ind, low_ind, vert_ind)]

    if len(triplets) == 0:
        print(f"Error!! No any pattern found for {name} !!   Img stored in {markup_errors}")
        cv.imwrite(f"{markup_errors}/NoTripl-{name}.png", img)
        return -1, None
    elif len(triplets) > 1:
        print(f"Error!! Too many patterns.  Img stored in {markup_errors}")
        for t in triplets:
            print(f"top: {t[0]}, low: {t[1]}, vert: {t[2]}")
        draw_rect(img.copy(), sticks, name)
        cv.imwrite(f"{markup_errors}/ManyTripl-{name}.png", img)
        return -1, None

    # only 1 triplet found
    top, low, vert = sticks[triplets[0][0]], sticks[triplets[0][1]], sticks[triplets[0][2]]

    # starting point (between cross-points of vertical stick with each horizontal stick
    cross_hv = Util.line_intersection(Util.rect_2points(top), Util.rect_2points(vert))
    cross_lv = Util.line_intersection(Util.rect_2points(low), Util.rect_2points(vert))
    start_point = Util.middle(cross_hv, cross_lv)

    # aim_line: angle is average (high,low) angle, go through start_point
    # (pleft, pright) - points where aim_line is crossing left and right borders
    aim_line_angle = (Util.get_angle(pts=Util.rect_2points(top)) + Util.get_angle(pts=Util.rect_2points(low))) / 2.
    dleft_w, dright_w = -start_point[0], 1920 - start_point[0]  # dw to left and right borders
    dleft_h = dleft_w * np.tan(np.deg2rad(aim_line_angle))
    dright_h = dright_w * np.tan(np.deg2rad(aim_line_angle))

    # store markup info for future draws
    markup['pleft'] = Util.int2((start_point[0] + dleft_w, start_point[1] + dleft_h))
    markup['pright'] = Util.int2((start_point[0] + dright_w, start_point[1] + dright_h))
    markup['start_point'] = Util.int2(start_point)
    markup['aim_line_angle'] = aim_line_angle
    markup['cross_hv'] = Util.int2(cross_hv)
    markup['cross_lv'] = Util.int2(cross_lv)
    markup['tl_dist'] = Util.dist(top[0], low[0])
    return 0, markup


def draw_rect(img, rect_lst, name="", color=(255, 0, 0)):
    img_copy = img  # .copy()
    for idx, rect in enumerate(rect_lst):
        # cv.drawContours(img_copy, contours, idx, (255, 255, 255), 1)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(img_copy, [box], 0, color, 2)
        cv.putText(img_copy, f"{idx} ", tuple(box[0]),  # {rect[1][0]:.0f} {rect[1][1]:.0f} {rect[2]:.0f}
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Util.show_img(img_copy, name)


# ------------------------------------------

def main():
    # get stat from video for finding markup

    results = []
    ok_cnt = 0

    # fs = FrameStream('img/tst/*.png')
    # fs = FrameStream('img/err/NoTr*_3275.png')
    fs = FrameStream('video/out2.avi')

    while True:

        img, fname, frame_cnt = fs.next_frame()
        if img is None:
            break
        if debug:
            print(f"{fname=}")

        # mark_up = MarkUp(img, os.path.basename(fname)[0:-4])
        mark_up = MarkUp(img, fname)

        if mark_up.markup is not None:
            results += [(fname, "OK")]
            ok_cnt += 1
            mark_up.show_markup()
        else:
            results += [(fname, "Fail!")]

        # Util.show_img(img, "out_img")

    for r in results:
        print(f"{r}")
    print(
        f"Found {ok_cnt} of {fs.frame_cnt}. Avg time/frame = {1 / fs.fps():.2f} s  FPS = {fs.fps():.1f}")


if __name__ == '__main__':
    main()

# if markup is None:
#     markup = MarkUp(img, fname)
#     markup.draw_markup(img)
#     Util.show_img(img,"markup set")
# else:
#     markup.draw_markup(img)
# Util.show_img(img, "out_img")
