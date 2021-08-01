# player_processed (use frame_processor from external file)
# modes: frame-by-frame, write, markup
# keys:
#   ' ' - frame mode, 'g' - go (stream),
#   's' - shot
#   'z' - on/off zone  drawing
#   left-mouse - add corner to zone, right-mose - reset zone
#   '1'-'9' - delays

import logging
import datetime
import time

import cv2 as cv
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from my_util import FrameStream, WriteStream
from swing_cutter import FrameProcessor  # delete if not need external FrameProc (internal dummy stub will be used instead)

MONITOR_MODE = True
# INPUT_SOURCE = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
INPUT_SOURCE = 'video/phone-range-2.mp4'  # 0.avi b2_cut phone-profil-evening-1.mp4 fac-2 nb-profil-1 (daylight) phone-range-2.mp4
# INPUT_SOURCE = '/run/user/1000/gvfs/mtp:host=Xiaomi_Redmi_Note_8_Pro_fukvv87l8pbuo7eq/Internal shared storage/DCIM/Camera/tst2.mp4'

REPEAT_MODE = False if not MONITOR_MODE else True

WRITE_MODE = True if not MONITOR_MODE and INPUT_SOURCE[0:4] == 'rtsp' else False
OUT_FILE_NAME = 'video/out2.avi'
WRITE_FPS = 25

FRAME_MODE_INITIAL = False
ZONE_DRAW_INITIAL = True
DELAY = 5  # delay in normal 'g'-mode


def main():
    logging.debug(f"\n\n\n\n\nPlayer started: {MONITOR_MODE=} {WRITE_MODE=} {OUT_FILE_NAME=} ")
    cv.namedWindow('out_frame')

    if not MONITOR_MODE:
        player(INPUT_SOURCE)

    else:  # MONITOR_MODE = True
        WatchDog.set_watchdog()
        while not WatchDog.new_file_arrived:
            time.sleep(1)
        print(f"first file arrived: {WatchDog.new_file_name}")
        player(WatchDog.new_file_name)

    cv.destroyAllWindows()


def player(input_source: str):
    frame_proc = FrameProcessor(input_source, win_name='out_frame')
    frame_mode = FRAME_MODE_INITIAL
    zone_draw_mode = ZONE_DRAW_INITIAL  # True - draw active zone (corners_lst) on all images

    while True:
        print(f"New file is playing: {input_source}")
        input_fs = FrameStream(input_source)
        out_fs = WriteStream(OUT_FILE_NAME, fps=WRITE_FPS) if WRITE_MODE else None

        while True:
            frame, frame_name, frame_cnt = input_fs.next_frame()
            if frame is None:
                break

            out_frame = frame_proc.process_frame(frame, frame_cnt, zone_draw_mode)

            cv.putText(out_frame, f"{frame_cnt}", (5, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if WRITE_MODE:
                out_fs.write(out_frame)

            cv.imshow(f'out_frame', out_frame)
            ch = cv.waitKey(0 if frame_mode else DELAY)
            if WatchDog.new_file_arrived:
                break
            if ch == ord('q'):
                return
            elif ch == ord('g'):
                frame_mode = False
                continue
            elif ch == ord(' '):
                frame_mode = True
                continue
            elif ch == ord('s'):
                snap_file_name = f'img/snap_{datetime.datetime.now().strftime("%d%m%Y_%H%M%S")}.png'
                cv.imwrite(snap_file_name, out_frame)
                continue
            elif ch == ord('z'):
                zone_draw_mode = not zone_draw_mode

        del input_fs
        if WRITE_MODE:
            del out_fs

        if WatchDog.new_file_arrived:
            input_source = WatchDog.new_file_name
            WatchDog.new_file_arrived = False

        # print(f"Finish. Duration={input_fs.total_time():.0f} sec, {input_fs.frame_cnt} frames,  fps={input_fs.fps():.1f} f/s")


class WatchDog:
    new_file_arrived: bool = False
    new_file_name = None

    @staticmethod
    def on_created(event):
        WatchDog.new_file_arrived = True
        WatchDog.new_file_name = event.src_path
        print(f"hey, {event.src_path} has been created!")

    @staticmethod
    def set_watchdog():
        my_event_handler = PatternMatchingEventHandler(['*'], None, True, True)

        my_event_handler.on_created = WatchDog.on_created

        my_observer = Observer()
        my_observer.schedule(my_event_handler, "video/swings/", recursive=False)

        my_observer.start()


#
# if "FrameProcessor" not in globals():
#     class FrameProcessorDummy:  # dummy, if not going to import external FrameProcessor
#         def __init__(self, file_name=None, win_name=None):
#             self.processor_name = "dummy"
#             pass
#
#         def process_frame(self, frame, frame_cnt, zone_draw_mode=False):
#             return frame
#
#         def end_stream(self, frame_cnt):
#             pass

if __name__ == '__main__':
    main()
