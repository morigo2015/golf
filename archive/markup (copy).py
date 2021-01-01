import math
import numpy as np
import cv2 as cv

inp_fname = "img/inp-2"
out_fname = "img/res"

debug = True
# debug = False

class MarkUp:
    _start_point = None
    _aim_line_angle = None
    _sticks = None
    _cross_hv = None
    _cross_lv = None

    def __init__(self, img):
        if debug:
            show_img(img, "_init")

        blur = cv.medianBlur(img, 11)
        if debug:
            show_img(blur, "_median_blur")

        low = (0, 0, 0)
        high = (255, 0, 255)
        mask_green = cv.inRange(blur, np.array(low, dtype="uint8"), np.array(high, dtype="uint8"))
        if debug:
            show_img(mask_green, "mask_green")

        mask = cv.dilate(mask_green, np.ones((5, 5), np.uint8), iterations=3)
        if debug:
            show_img(mask, "mask")

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rect_lst = [(idx, cv.minAreaRect(contours[idx])) for idx, cnt in enumerate(contours) if
                    cv.contourArea(cnt) > 5000]
        long_rect_lst = [(idx, rect, self._rect_2points(rect)) for idx, rect in rect_lst if max(rect[1]) / min(rect[1]) > 10]
        if debug:
            show_rect_lst(img, long_rect_lst, contours, "long_rects")

        if len(long_rect_lst) != 3:
            print(f"Error!! There must be 3 sticks in frame. Found {len(long_rect_lst)} items")
            exit(0)

        rect_sorted = sorted(long_rect_lst, key=lambda r: r[1][0][1])  # sort by increasing center_h
        self._sticks = {'high': rect_sorted[0], 'low': rect_sorted[1], 'vert': rect_sorted[2]}
        if debug:
            show_sticks(img,self._sticks,"sticks")

        self._field_markup(img, self._sticks)


    def draw(self, image):
        # draw_markup
        cv.line(image, self.p1, self.p2, (0, 255, 0), 1)  # aiming line
        hv = int(self._cross_hv[0]), int(self._cross_hv[1])
        lv = int(self._cross_lv[0]), int(self._cross_lv[1])
        cv.line(image, hv, lv, (0, 255, 0), 2)
        cv.circle(image, self._start_point, 3, (0, 255, 0), 1)  # starting cross

    def _field_markup(self, img, sticks_rect):
        # find sp
        cross_hv = self._line_intersection(sticks_rect['high'][2], sticks_rect['vert'][2])
        cross_lv = self._line_intersection(sticks_rect['low'][2], sticks_rect['vert'][2])
        start_point = (cross_lv[0] + cross_hv[0]) / 2, (cross_lv[1] + cross_hv[1]) / 2

        # find aim_line: average (high,low) angle, go through start_point
        # (p1, p2) - crossing points for aim_line and (left, right) borders
        aim_line_angle = (sticks_rect['high'][1][2] + sticks_rect['low'][1][2]) / 2
        p1_w, p2_w = 0, 1920
        sp_w, sp_h = start_point
        d1_w, d2_w = p1_w - sp_w, p2_w - sp_w
        d1_h = d1_w * np.tan(np.deg2rad(90. + aim_line_angle))
        d2_h = d2_w * np.tan(np.deg2rad(90. + aim_line_angle))
        p1_h, p2_h = sp_h + d1_h, sp_h + d2_h
        p1 = int(p1_w), int(p1_h)
        p2 = int(p2_w), int(p2_h)
        start_point = tuple(map(int, start_point))

        self.p1, self.p2 = p1, p2
        self._start_point = start_point
        self._aim_line_angle = aim_line_angle
        self._cross_hv, self._cross_lv = cross_hv, cross_lv
        return


    def _line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def _rect_2points(self,rect):
        box = cv.boxPoints(rect)
        dist01 = math.sqrt((box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2)
        dist12 = math.sqrt((box[2][0] - box[1][0]) ** 2 + (box[2][1] - box[1][1]) ** 2)
        if dist01 > dist12:
            if debug:
                print(f"проворачиваем точки: {box}")
            box = [box[1], box[2], box[3], box[0]]  # rotate points to ensure box[0]-box[1] is short side of rect
        p1 = (box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2
        p2 = (box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2
        if debug:
            print(f"{box=} {rect=} {dist01=:.2f} {dist12=:.2f} {p1=} {p2=}")
        return p1, p2


# -------------------------------------------------------------------------------------------------------------

def show_rect_lst(img, rect_lst, contours, name=""):
    img_copy = img.copy()
    for idx, rect, _ in rect_lst:
        cv.drawContours(img_copy, contours, idx, (255, 255, 255), 1)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(img_copy, [box], 0, (255, 0, 0), 2)
        # cv.putText(img_copy, f"{idx} {rect[1][0]:.0f} {rect[1][1]:.0f} {rect[2]:.0f}", tuple(box[0]),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if debug:
            print(f"{idx=} {box=} {rect=}")
    show_img(img_copy, name)

def show_sticks(img, sticks, name=""):
    img_copy = img.copy()
    for s in sticks:
        stick = sticks[s]
        idx = stick[0]
        rect = stick[1]
        p = stick[2]
        print(f"{s=}{idx=} ang={rect[2]:.2f}") # {p[0]=:.0f}{p[1]=:.0f}
        cv.line(img_copy,(int(p[0][0]),(int(p[0][1]))),(int(p[1][0]),(int(p[1][1]))),(255,0,0),1)
        cv.putText(img_copy, f"{s}{idx} {rect[1][0]:.0f} {rect[1][1]:.0f} {rect[2]:.0f}", (int(p[0][0]),(int(p[0][1]))),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    show_img(img_copy, name)

def show_img(img, name=""):
    cv.imshow(f"{name}", img)
    ch = cv.waitKey(0)
    if ch == ord('s'):
        cv.imwrite(f"{out_fname}{name}.png", img)
    elif ch == ord('q'):
        exit(0)

def main():
    img = cv.imread(f"{inp_fname}.png")

    mark_up = MarkUp(img)
    mark_up.draw(img)
    show_img(img, "out_img")

if __name__ == '__main__':
    main()

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
