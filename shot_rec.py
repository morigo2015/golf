# shot_rec.py: shot recorder
# extract shots from input video

import cv2 as cv

from util import Util
from collections import deque

debug = True


# debug = False

class FrameDescriptor:
    def __init__(self, frame, frame_cnt):
        self.frame = frame.copy()
        self.frame_cnt = frame_cnt


class FrameBuff:
    buff_size = 100
    fd_buffer = deque()

    @classmethod
    def save_bg(cls,frame):
        cls.bg = frame

    @classmethod
    def add_frame(cls,frame, frame_cnt):
        fd = FrameDescriptor(frame, frame_cnt)
        cls.fd_buffer.append(fd)

    @classmethod
    def next_frame(cls, frame, frame_cnt):
        if frame_cnt == 1:
            cls.save_bg(frame)

        if len(cls.fd_buffer) < cls.buff_size:
            cls.add_frame(frame,frame_cnt)



def main():
    pass


if __name__ == '__main__':
    main()
