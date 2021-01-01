# shot_rec.py: shot recorder
# extract shots from input video

import cv2 as cv

from markup import MarkUp
from util import Util

debug = True
# debug = False

inp_fname = "img/tst/tst-img-1"

class ShotRec:
    def __init__(self, markup: MarkUp):
        self.markup = markup
        self.start_point = markup.get_start_point()  # (w,h)
        self.start_area = markup.get_start_area()  # (l,r,u,d)
        self.ball_status = 'unknown'  # 'unknown' 'present', 'absent'

    def input_frame(self, frame):  # eat input frames
        l, r, u, d = self.start_area
        ball_area = frame[l:r, u:d]
        if debug:
            Util.show_img(ball_area, "ball_area")
        
    def draw_shot_status(self, frame):  # add shot status info to frame
        return frame


def main():
    img = cv.imread(f"{inp_fname}.png")
    mark_up = MarkUp(img)
    mark_up.draw(img)
    Util.show_img(img, "out_img")
    shot_rec = ShotRec(mark_up)
    shot_rec.input_frame(img)


if __name__ == '__main__':
    main()