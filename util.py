import math
import numpy as np
import cv2 as cv


class Util:
    out_fname = "img/res"

    @staticmethod
    def int2(p):
        return int(p[0]), int(p[1])

    @staticmethod
    def get_angle(p1, p2):
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
    def show_img(img, name=""):
        cv.imshow(f"{name}", img)
        ch = cv.waitKey(0)
        if ch == ord('s'):
            cv.imwrite(f"{Util.out_fname}{name}.png", img)
        elif ch == ord('q'):
            exit(0)


# -------------------------------------------------------------------------------------------------------------

def main():
    print(f"{Util.dist((100, 100), (200, 200))}")



if __name__ == '__main__':
    main()
