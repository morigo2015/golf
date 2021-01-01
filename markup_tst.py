import math
import numpy as np
import cv2 as cv

from util import Util

inp_fname = "img/tst/tst-img-3"

debug = True


# debug = False


class MarkUp:
    _sticks = None  # list of 3 sticks reverse ordered by distance to golfer (increasing h): [(idx, (center,size,angle))]
    _markup = None  # high,low,vert, start_point, aim_line_angle, cross_hv, cross_lv

    def __init__(self, img):

        if img is None:
            print(f"Error!! No input image")
            exit(-1)
        if img.shape != (1080, 1920, 3):
            print(f"Error!! Illegal image. Shape={img.shape} instead of (1080,1920,3)")
            exit(-1)
        self._start_img = img
        if debug:
            Util.show_img(img, "_start_img")

        self._sticks = _get_sticks(img)
        self._markup = _get_markup(self._sticks)
        set_start_area(self._markup)
        if debug:
            self._show_markup()

    def draw(self, image):
        # draw
        cv.line(image, self._markup['pleft'], self._markup['pright'], (0, 50, 0), 1)  # aiming line
        cv.line(image, self._markup['cross_hv'], self._markup['cross_lv'], (0, 50, 0), 2)
        cv.circle(image, self._markup['start_point'], 3, (0, 50, 0), 1)  # starting cross
        return image

    def get_start_point(self):
        return self._markup['start_point']

    def get_start_area(self):
        return self._markup['start_area']

    def _show_markup(self):
        img = self._start_img.copy()
        self.draw(img)
        Util.show_img(img, "draw markup")


def _get_sticks(img):
    # image processing
    # blur -> mask green -> dilate -> get contours -> select (area>5000,length/width>10) -> sort by center_h

    img = cv.medianBlur(img, 11)

    # img = cv.bilateralFilter(img, 5, 175, 175)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # lower_color = np.array([25, 157, 8])
    # upper_color = np.array([48, 255, 48])
    lower_color = np.array([41, 120, 5])
    upper_color = np.array([58, 255, 40])
    mask = cv.inRange(img_hsv, lower_color, upper_color)

    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((7, 7), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, np.ones((7, 7), np.uint8))

    # applied_mask = cv.bitwise_and(img_hsv, img_hsv, mask=mask)

    """
    blur = cv.medianBlur(img, 11)
    if debug:
        Util.show_img(blur, "_median_blur")

    blur = cv.bilateralFilter(img, 30, 250, 100)
    if debug:
        Util.show_img(blur, "_bilateral_blur")

    low = (0, 0, 0)
    high = (20, 20, 20)  # 0 255 0
    mask_green = cv.inRange(blur, np.array(low, dtype="uint8"), np.array(high, dtype="uint8"))
    if debug:
        Util.show_img(mask_green, "mask_green")

    
    # Convert BGR to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower = np.array([0,70,50])
    upper = np.array([170,180,255])
    # Threshold the HSV image to get only blue colors
    mask_green = cv.inRange(hsv, np.array(lower,dtype="uint8"), np.array(upper,dtype="uint8"))
    if debug:
        Util.show_img(mask_green, "mask_green_hsv")
    """

    # mask = cv.dilate(mask_green, np.ones((5, 5), np.uint8), iterations=5)  # 3

    if debug:
        Util.show_img(mask, "mask")

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rect_desc_lst = [(idx, cv.minAreaRect(contours[idx])) for idx, cnt in enumerate(contours)
                     if cv.contourArea(cnt) > 3000]  # 5000
    if debug:
        show_rect_desc(img, rect_desc_lst, contours, "_rect_desc_area")
    rect_desc_lst = [rd for rd in rect_desc_lst
                     if max(rd[1][1]) / min(rd[1][1]) > 8]  # rd[1][1] - size tuple # 10
    if debug:
        show_rect_desc(img, rect_desc_lst, contours, "_rect_desc_size")

    if len(rect_desc_lst) != 3:
        print(f"Error!! Must be 3 sticks. Found {len(rect_desc_lst)} items. See file:{Util.out_fname}_nomarkup.png")
        cv.imwrite(f"{Util.out_fname}_nomarkup.png", img)
        raise ValueError(len(rect_desc_lst))

    rect_desc_lst = sorted(rect_desc_lst, key=lambda r: r[1][0][1])  # sort by increasing center_h
    return rect_desc_lst


def _get_markup(sticks_lst):
    _markup = {}

    # sticks (2 horizontal, 1 vertical)
    for mark, rect_desc in zip(['high', 'low', 'vert'], sticks_lst):
        p1, p2 = Util.rect_2points(rect_desc[1])
        angle = Util.get_angle(p1, p2)
        idx = rect_desc[0]
        _markup[mark] = (p1, p2, angle, idx)

    # starting point (between cross-points of vertical stick with each horizontal stick
    cross_hv = Util.line_intersection((_markup['high'][0], _markup['high'][1]),
                                      (_markup['vert'][0], _markup['vert'][1]))
    cross_lv = Util.line_intersection((_markup['low'][0], _markup['low'][1]), (_markup['vert'][0], _markup['vert'][1]))
    start_point = Util.middle(cross_hv, cross_lv)

    # aim_line: angle is average (high,low) angle, go through start_point
    # (pleft, pright) - points where aim_line is crossing left and right borders
    aim_line_angle = (_markup['high'][2] + _markup['low'][2]) / 2.
    dleft_w, dright_w = -start_point[0], 1920 - start_point[0]  # dw to left and right borders
    dleft_h = dleft_w * np.tan(np.deg2rad(aim_line_angle))
    dright_h = dright_w * np.tan(np.deg2rad(aim_line_angle))

    # store markup info for future draws
    _markup['pleft'] = Util.int2((start_point[0] + dleft_w, start_point[1] + dleft_h))
    _markup['pright'] = Util.int2((start_point[0] + dright_w, start_point[1] + dright_h))
    _markup['start_point'] = Util.int2(start_point)
    _markup['aim_line_angle'] = aim_line_angle
    _markup['cross_hv'] = Util.int2(cross_hv)
    _markup['cross_lv'] = Util.int2(cross_lv)
    return _markup


def set_start_area(markup):
    sp = markup['start_point']
    markup['start_area'] = (sp[0] - 150, sp[0] + 50, sp[1] - 50, sp[1] + 50)  # (left,right,up,down) extended to left


def show_rect_desc(img, rect_desc_lst, contours, name=""):
    img_copy = img # .copy()
    for idx, rect in rect_desc_lst:
        cv.drawContours(img_copy, contours, idx, (255, 255, 255), 1)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(img_copy, [box], 0, (255, 0, 0), 2)
        cv.putText(img_copy, f"{idx} ", tuple(box[0]),  # {rect[1][0]:.0f} {rect[1][1]:.0f} {rect[2]:.0f}
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # print(f"{idx=} {box=} {rect=}")
    # Util.show_img(img_copy, name)


import glob


def main():
    results = {}
    for fname in sorted(glob.glob('img/tst_sticks/*.png')):
        print(f"{fname=}")
        # img = cv.imread(f"{inp_fname}.png")
        img = cv.imread(fname)

        img = cv.medianBlur(img, 5)
        grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        grey = cv.medianBlur(grey, 5)
        # Util.show_img(grey, "grey")

        # ret2, mask = cv.threshold(grey, 0, 120, cv.THRESH_BINARY_INV)
        # mask = np.bitwise_and(10 < grey, grey < 20)
        # mask = mask.astype(np.uint8) * 255

        lim1, lim2, lim3 = 3, 10, 20
        mask1 = grey < lim1
        mask2 = grey < lim2
        mask3 = grey < lim3
        # mask2 = np.bitwise_and(lim1 <= grey, grey < lim2)
        # mask3 = np.bitwise_and(lim2 <= grey, grey < lim3)

        def show_mask(mask, name=""):
            mask = mask.astype(np.uint8) * 255
            mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)  # 3
            # mask = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=9)
            # cv.imshow(name, mask)

            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            rect_desc_lst = [(idx, cv.minAreaRect(contours[idx])) for idx, cnt in enumerate(contours)
                             if cv.contourArea(cnt) > 3000]  # 5000
            rect_desc_lst = [rd for rd in rect_desc_lst
                             if max(rd[1][1]) / min(rd[1][1]) > 6]  # rd[1][1] - size tuple # 10
            if debug:
                show_rect_desc(img, rect_desc_lst, contours, "_rect_desc_area")

            print(f"{name} len={len(rect_desc_lst)}")


        show_mask(mask1,"mask1")
        show_mask(mask2,"mask2")
        show_mask(mask3,"mask3")
        cv.imshow(fname,img)
        cv.waitKey(0)

        # mask = cv.adaptiveThreshold(grey, 120, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11,2 )
        # Util.show_img(mask, "mask")

        # mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)  # 3
        # mask = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=5)
        # mask = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((7, 7), np.uint8))
        # mask = cv.morphologyEx(mask, cv.MORPH_ERODE, np.ones((7, 7), np.uint8))
        # Util.show_img(mask, f"{fname}-mask")
    exit(0)

    #
    #     try:
    #         mark_up = MarkUp(img)
    #     except ValueError as err:
    #         results[fname] = err.args[0]
    #     else:
    #         results[fname] = 'OK'
    #
    #     # mark_up.draw(img)
    #     # Util.show_img(img, "out_img")
    #
    # err_files = [(r, results[r]) for r in results if results[r] != 'OK']
    # print(f"Ok - {len(results) - len(err_files)}, err - {len(err_files)} of {len(results)}")
    # for r in results:
    #     print(f"{r} - {results[r]}")


if __name__ == '__main__':
    main()
