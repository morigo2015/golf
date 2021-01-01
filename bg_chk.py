import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from util import Util

debug = True


# debug = False

# def _get_sticks(img):
# image processing
# blur -> mask green -> dilate -> get contours -> select (area>5000,length/width>10) -> sort by center_h

# blur = cv.medianBlur(img, 11)
# if debug:
#     Util.show_img(blur, "_median_blur")

# blur = cv.bilateralFilter(img, 30, 250, 100)
# if debug:
#     Util.show_img(blur, "_bilateral_blur")

# low = (0, 0, 0)
# high = (0, 255, 0)
# mask_green = cv.inRange(blur, np.array(low, dtype="uint8"), np.array(high, dtype="uint8"))
# if debug:
#     Util.show_img(mask_green, "mask_green")

#
# get_stat(img, "origin")
# # Util.show_img(img,"init")


# Util.show_img(img,"med blur")

def get_stat(img, name):
    m = {}  # 2d array of pix
    pix = {}  # 1d array of pix
    stat = {}  # channel stat
    for ch in [0, 1, 2]:
        m[ch] = img[..., ch]
        pix[ch] = m[ch].reshape(m[ch].shape[0] * m[ch].shape[1])
        stat[ch] = (min(pix[ch]), max(pix[ch]))
    stat_str = f"[{stat[0]} : {stat[1]} : {stat[2]}]"
    cv.putText(img, f"{stat_str}", (35, 35),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    print(f"    {name}: {stat_str}")

    return stat_str


import glob


def main():
    results = {}
    for fname in sorted(glob.glob('img/tst/tst-img-3.png')):  # img/bg/bg1.png
        print(f"fname = {fname}")
        img_init = cv.imread(fname)

        stat = []
        while True:

            img = img_init.copy()

            # Select ROI
            r = cv.selectROI(img)
            if r == (0, 0, 0, 0):
                break

            # Crop image
            imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            imCrop = cv.medianBlur(imCrop, 15)

            img = cv.cvtColor(imCrop, cv.COLOR_BGR2HSV)

            stat.append([(np.min(ch), np.max(ch)) for ch in cv.split(img)])

            # get_stat(img, "hsv")

            # color = ('b', 'g', 'r')
            # for i, col in enumerate(color):
            #     histr = cv.calcHist([img], [i], None, [256], [0, 256])
            #     plt.plot(histr, color=col)
            #     plt.xlim([0, 256])
            # plt.show()

        for s in stat:
            print(s)

        limits = []
        for i, ch in enumerate(['h', 's', 'v']):
            min_lst = [s[i][0] for s in stat]
            max_lst = [s[i][1] for s in stat]
            intersec = (max(min_lst), min(max_lst))  # intersection of ranges for current channel
            union = (min(min_lst), max(max_lst))  # union of ranges for current channel
            IoU = (intersec[1] - intersec[0]) / (union[1] - union[0])
            print(f"{ch}: intersec={intersec}, union={union}, IoU = {IoU:.2f} ")
            limits.append({'intersec': intersec, 'union': union})
        print(f"{limits=}")

        img = img_init.copy()
        img = cv.medianBlur(img, 15)
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        low_intr = np.array([lim['intersec'][0] for lim in limits])
        upp_intr = np.array([lim['intersec'][1] for lim in limits])
        low_union = np.array([lim['union'][0] for lim in limits])
        upp_union = np.array([lim['union'][1] for lim in limits])
        print(f"{low_intr=}\n{upp_intr=}\n{low_union=}\n{upp_union=}")

        # lower_color = np.array([25, 157, 8])
        # upper_color = np.array([48, 255, 48])
        mask_intersec = cv.inRange(img, low_intr, upp_intr)
        mask_union = cv.inRange(img, low_union, upp_union)

        cv.imshow("intersec",mask_intersec)
        cv.imshow("union",mask_union)

        # lines
        mask = cv.bitwise_not(mask_union)
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((7, 7), np.uint8))
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, np.ones((7, 7), np.uint8))
        cv.imshow("mask",mask)

        img = img_init.copy()
        lines = cv.HoughLines(mask, 1, np.pi / 180, 2000)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.imshow('green(union)- hough', img)

        cv.waitKey(0)


if __name__ == '__main__':
    main()
