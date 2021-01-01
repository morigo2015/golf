# player
# modes: frame-by-frame, write, markup
# keys: g,q,space,s,

import time
import datetime
import numpy as np
import cv2 as cv

from markup import MarkUp

# video_file_name = 'output.avi'
# video_file_name = 'rejected.avi'
# video_file_name = 'noise.avi'
# video_file_name = 'short2.mp4'
video_file_name = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
# video_file_name = 'video/out2.avi'

write_mode = True
# write_mode = False

# markup_mode = True
markup_mode = False

out_file_name = 'video/out2.avi'
frame_mode_initial = False
# frame_mode_initial = True
delay_initial = 1
delay_multiplier = 60

def main():
    frame_mode = frame_mode_initial
    delay = delay_initial
    cap = cv.VideoCapture(video_file_name)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    frame_cnt = 0

    if write_mode:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(out_file_name, fourcc, 20.0, (1920, 1080))

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_cnt = frame_cnt + 1

        out_frame = process_frame(frame, frame_cnt)
        cv.putText(out_frame, f"D:{delay / delay_multiplier:.0f}/{frame_cnt}", (5, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.imshow('frame', out_frame)

        if write_mode:
            out.write(out_frame)

        ch = cv.waitKey(0 if frame_mode else delay)
        if ch == ord('q'):
            break
        elif ch == ord('g'):
            frame_mode = False
            continue
        elif ch == ord(' '):
            frame_mode = True
            continue
        elif ch == ord('s'):
            dt = datetime.datetime.now()
            snap_fname = f'img/snap_{dt.strftime("%d%m%Y_%H%M%S")}.png'
            cv.imwrite(snap_fname, out_frame)
            continue
        elif ord('0') < ch <= ord('9'):
            delay = delay_multiplier * (ch - ord('0'))
            frame_mode = False
            print(f"{delay=}")

    end_time = time.time()
    print(
        f"Finish. Duration={end_time - start_time:.0f} sec, {frame_cnt=} fps={frame_cnt / (end_time - start_time):.1f}")

    cap.release()
    if write_mode:
        out.release()
    cv.destroyAllWindows()


markup: MarkUp


def process_frame(frame, frame_cnt):
    global markup
    if markup_mode:
        if frame_cnt == 1:
            markup = MarkUp(frame)
        else:
            markup.draw(frame)

    return frame


if __name__ == '__main__':
    main()
