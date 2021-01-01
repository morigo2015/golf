# m2.py
# player: frame-by-frame,
# g,q,space

import time
import numpy as np
import cv2 as cv

from markup_old import MarkUp

# inp_source_name = 'video/out2.avi'
video_file_name = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
# inp_source_name = 'rejected.avi'
# inp_source_name = 'noise.avi'
# inp_source_name = 'short2.mp4'

frame_mode_initial = True
# frame_mode_initial = False


def main():
    frame_mode = frame_mode_initial
    cap = cv.VideoCapture(video_file_name)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    frame_cnt = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_cnt = frame_cnt + 1

        if frame_cnt == 1:
            markup = MarkUp(frame)
        else:
            markup.draw(frame)

        out_frame = frame_processing(frame, frame_cnt)
        cv.imshow('frame', out_frame)

        ch = cv.waitKey(0 if frame_mode else 1)
        if ch == ord('q'):
            break
        elif ch == ord('g'):
            frame_mode = False
            continue
        elif ch == ord(' '):
            frame_mode = True
            continue

    end_time = time.time()
    print(
        f"Finish. Duration={end_time - start_time:.0f} sec, {frame_cnt=} fps={frame_cnt / (end_time - start_time):.1f}")

    cap.release()
    cv.destroyAllWindows()


def frame_processing(frame, frame_cnt):
    return frame


if __name__ == '__main__':
    main()
