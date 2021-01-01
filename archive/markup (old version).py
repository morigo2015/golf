import math
import numpy as np
import cv2 as cv

from util import Util

inp_fname = "img/tst/tst-img-7"

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
        cv.line(image, self._markup['pleft'], self._markup['pright'], (0, 0, 50), 1)  # aiming line
        cv.line(image, self._markup['cross_hv'], self._markup['cross_lv'], (0, 0, 50), 2)
        cv.circle(image, self._markup['start_point'], 3, (0, 0, 50), 1)  # starting cross
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

    blur = cv.medianBlur(img, 11)
    if debug:
        Util.show_img(blur, "_median_blur")

    low = (0, 0, 0)
    high = (0, 255, 0)
    mask_green = cv.inRange(blur, np.array(low, dtype="uint8"), np.array(high, dtype="uint8"))
    if debug:
        Util.show_img(mask_green, "mask_green")

    mask = cv.dilate(mask_green, np.ones((5, 5), np.uint8), iterations=5)  # 3
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
    cross_hv = Util.line_intersection((_markup['high'][0], _markup['high'][1]), (_markup['vert'][0], _markup['vert'][1]))
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
    img_copy = img.copy()
    for idx, rect in rect_desc_lst:
        cv.drawContours(img_copy, contours, idx, (255, 255, 255), 1)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(img_copy, [box], 0, (255, 0, 0), 2)
        cv.putText(img_copy, f"{idx} ", tuple(box[0]),  # {rect[1][0]:.0f} {rect[1][1]:.0f} {rect[2]:.0f}
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # print(f"{idx=} {box=} {rect=}")
    Util.show_img(img_copy, name)

import glob

def main():
    results={}
    for fname in sorted( glob.glob('img/tst_sticks/7.png') ):
        # img = cv.imread(f"{inp_fname}.png")
        img = cv.imread(fname)

        try:
            mark_up = MarkUp(img)
        except ValueError as err:
            results[fname] = err.args[0]
        else:
            results[fname] = 'OK'

        mark_up.draw(img)
        Util.show_img(img, "out_img")

    err_files = [(r,results[r]) for r in results if results[r]!='OK']
    print(f"Ok - {len(results)-len(err_files)}, err - {len(err_files)} of {len(results)}")
    for r in results:
        print(f"{r} - {results[r]}")

if __name__ == '__main__':
    main()

    #
    # def show_sticks(img, sticks, name=""):
    #     img_copy = img.copy()
    #     for s in sticks:
    #         stick = sticks[s]
    #         idx = stick[0]
    #         rect = stick[1]
    #         p = stick[2]
    #         print(f"{s=}{idx=} ang={rect[2]:.2f}")  # {p[0]=:.0f}{p[1]=:.0f}
    #         cv.line(img_copy, (int(p[0][0]), (int(p[0][1]))), (int(p[1][0]), (int(p[1][1]))), (255, 0, 0), 1)
    #         cv.putText(img_copy, f"{s}{idx} {rect[1][0]:.0f} {rect[1][1]:.0f} {rect[2]:.0f}",
    #                    (int(p[0][0]), (int(p[0][1]))),
    #                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    #     show_img(img_copy, name)

    # blur = cv.medianBlur(img, 11)
    #
    # low = (0, 0, 0)
    # high = (255, 0, 255)
    # mask_green = cv.inRange(blur, np.array(low, dtype="uint8"), np.array(high, dtype="uint8"))
    #
    # mask = cv.dilate(mask_green, np.ones((5, 5), np.uint8), iterations=3)
    #
    # contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # rect_lst = [(idx, cv.minAreaRect(contours[idx])) for idx, cnt in enumerate(contours) if cv.contourArea(cnt) > 5000]
    # long_rect_lst = [(idx, rect, rect_2points(rect)) for idx, rect in rect_lst if max(rect[1]) / min(rect[1]) > 10]
    #
    # rect_sorted = sorted(long_rect_lst, key=lambda r: r[1][0][1])  # sort by increasing center_h
    # sticks_rect = {'high': rect_sorted[0], 'low': rect_sorted[1], 'vert': rect_sorted[2]}
    #
    # start_point, aim_line_angle = field_markup(img, sticks_rect)
    # if debug:
    #     show_img(img,"_init")
    #     show_img(blur,"_median_blur")
    #     show_img(mask_green, "mask_green")
    #     show_img(mask, "mask")
    #     show_rect_lst(img, long_rect_lst, contours, "long_rects")
    #     show_img(img, "markup")

    # def show_sticks(img, sticks, start_point, name=""):
    #     for s in sticks:
    #         stick = sticks[s]
    #         p1, p2 = stick[2][0], stick[2][1]
    #         p1 = tuple(map(int, p1))
    #         p2 = tuple(map(int, p2))
    #         cv.line(img, p1, p2, (255, 0, 255), 2)
    #     show_img(img, name)

    #
    # def stick_2points(center, angle):
    #     p1_w, p1_h = center
    #     p2_w = 1920 - 1
    #     dw = p2_w - p1_w
    #     dh = int(dw * np.tan((90. + angle) * np.pi / 180.))
    #     p2_h = int(center[1] + dh)
    #     if 0 <= p2_h <= 1080:
    #         print(f"normal case: {center=} {angle=} {dw=} {dh=} {p2_w=} {p2_h=} ")
    #         return p2_w, p2_h
    #     elif p2_h < 0:
    #         p2_h = 0
    #         p2_w = p1_w + int(p1_h * dw / dh)
    #         print(f"special case (p2_h<0): {center=} {angle=} {dw=} {dh=} {p2_w=} {p2_h=} ")
    #         return p2_w, p2_h
    #     else:  # p2_h > 1080
    #         print("!!!!! this case is not developed yet! ((")
    #         exit(0)
    #

    # def max_line(center, angle):
    #     p1 = center[0] - 500, h_line(center[0] - 50, center, angle)
    #     p2 = center[0] + 500, h_line(center[0] + 50, center, angle)
    #     return p1, p2
    #
    #
    # def h_line(w, center, angle):
    #     k = -np.tan(np.deg2rad(angle + 90.))
    #     b = -center[1] - center[0] * np.tan(np.deg2rad(angle + 90.))
    #     h = -int(k * w + b)
    #     if debug:
    #         print(f"{w=:.0f} {center=} {angle=:.2f} {k=:.2f} {b=:.2f} {h=:.0f}")
    #     return h
    #

    # def get_sticks(rect_lst):
    #     if len(rect_lst) != 3:
    #         print("Error!!: must be just 3 sticks\n{rect_lst=}")
    #         exit(0)
    #
    #     # item: (idx, rect=((center_w,center_h),(size_w,size_h),angle)
    #     for r in rect_lst: print(f"idx={r[0]} c={r[1][0]} size={r[1][1]} and={r[1][2]}")
    #
    #     rect_sorted = sorted(rect_lst, key=lambda r: r[1][0][1])  # sort by center_h
    #     print(f"sorted:")
    #     for r in rect_sorted: print(f"idx={r[0]} c={r[1][0]} size={r[1][1]} and={r[1][2]}")
    #
    #     return {'high': rect_2_stick(rect_sorted[0]),
    #             'low': rect_2_stick(rect_sorted[1]),
    #             'vert': rect_2_stick(rect_sorted[2])}
    #

    # cont_idx_lst = [idx for idx, cnt in enumerate(contours) if cv.contourArea(cnt) > 5000]

    # print([f"{idx=} area={cv.contourArea(contours[idx])}" for idx in cont_idx_lst])
    # for idx in cont_idx_lst:
    #     print(f"{idx=} {cv.contourArea(contours[idx])}")
    #     cv.drawContours(img_copy, contours, idx, (255, 255, 255), 2)
    # show_img(img_copy, "mask_contours")

    # mask_open = cv.morphologyEx(mask_green, cv.MORPH_OPEN, kernel)
    # show_img(mask_open, "mask_open")
    #
    # mask_close = cv.morphologyEx(mask_green, cv.MORPH_CLOSE, kernel)
    # show_img(mask_close, "mask_close")
    #
    # mask = cv.dilate(mask_green, kernel, iterations=5)
    # show_img(mask, "mask")
    # mask_erod = cv.erode(mask_green,  kernel, iterations=5)
    # show_img(mask_erod, "mask_erod")

    # res = cv.bitwise_and(blur, blur, mask_green = mask_green)
    # show_img(res,"blur_color")

    # gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # show_img(gray,"_blur_grey")
    #
    # gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray2 = cv.medianBlur(gray2,11)
    # show_img(gray2,"_grey_blur")
