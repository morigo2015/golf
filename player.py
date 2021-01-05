# player
# modes: frame-by-frame, write, markup
# keys: g,q,space,s,

import time
import datetime
import numpy as np
import cv2 as cv

from shot_finder import ShotFinder
from util import FrameStream

# inp_source_name = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
inp_source_name = 'video/out2.avi' #    video/phone-2(60).mp4
# inp_source_name = '/run/user/1000/gvfs/mtp:host=Xiaomi_Redmi_Note_8_Pro_fukvv87l8pbuo7eq/Internal shared storage/DCIM/Camera/tst2.mp4'

process_mode = True
# process_mode = False

write_mode = True if inp_source_name[0:4] == 'rtsp' else False
# write_mode = False

frame_mode_initial = False
# frame_mode_initial = True

delay_initial = 1
delay_multiplier = 60
out_file_name = 'video/out2.avi'


def main():
    frame_mode = frame_mode_initial
    delay = delay_initial
    fs = FrameStream(inp_source_name)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(out_file_name, fourcc, 20.0, (1920, 1080)) if write_mode else None

    while True:
        frame, frame_name, frame_cnt = fs.next_frame()
        if frame is None:
            break

        if process_mode:
            out_frame = process_frame(frame, frame_name, frame_cnt)
        else:
            out_frame = frame

        cv.putText(out_frame, f"D:{delay / delay_multiplier:.0f}/{frame_cnt}", (5, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.imshow(f'out_frame', out_frame)

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

    process_end_of_stream(frame_cnt)

    print(
        f"Finish. Duration={fs.total_time():.0f} sec, {fs.frame_cnt} frames,  fps={fs.fps():.1f} f/s")

    del fs
    if write_mode:
        out.release()
    cv.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------

from util import Util


def process_frame(frame, frame_name, frame_cnt):
    ShotFinder.next_frame(frame, frame_cnt)
    return frame
    # return frame # vanilla player
    #
    # global markup
    # if markup_mode:
    #     # print(f"{frame_cnt=}")
    #     if markup is None:
    #         markup = MarkUp(frame, frame_name)
    #         if markup is not None:
    #             print(f"markup set on {frame_cnt} in {frame_name}.\n{markup.markup=}")
    #             # markup.show_markup()
    #         else:
    #             print(f"markup is not set of first frame!")
    #             exit()
    #     else:
    #         start_area = markup.start_area
    #         if start_area is None:
    #             print(f"start area is not found!!")
    #             exit()
    #         # if 1 < frame_cnt < 234:
    #         #     return frame
    #         got_ball = start_area.got_ball(frame)
    #         if got_ball:
    #             start_area.draw_ball(frame)
    #         markup.draw_markup(frame)
    #         markup.start_area.draw_start_area(frame)
    # return frame


def process_end_of_stream(frame_cnt):
    pass
    # global markup
    # if markup is not None:
    #     markup.start_area.print_ball_stat()


if __name__ == '__main__':
    main()
