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
import os.path
import time

import cv2 as cv
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from my_util import FrameStream, WriteStream
from swing_cutter import FrameProcessor  # delete if not need external FrameProc (internal dummy stub will be used instead)


class Player:
    # INPUT_SOURCE = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
    # INPUT_SOURCE = 'video/phone-range-2.mp4'  # 0.avi b2_cut phone-profil-evening-1.mp4 fac-2 nb-profil-1 (daylight) phone-range-2.mp4
    # INPUT_SOURCE = '/run/user/1000/gvfs/mtp:host=Xiaomi_Redmi_Note_8_Pro_fukvv87l8pbuo7eq/Internal shared storage/DCIM/Camera/tst2.mp4'

    REPEAT_MODE = True

    FRAME_MODE_INITIAL = False
    ZONE_DRAW_INITIAL = True
    WIN_NAME = "Swing Player"
    WIN_XY = (-9999, 0)  # move to left

    def __init__(self):
        self.frame_mode = Player.FRAME_MODE_INITIAL
        self.zone_draw_mode = Player.ZONE_DRAW_INITIAL  # True - draw active zone (corners_lst) on all images
        self.input_source = None
        self.input_fs = None
        self.delay = 25
        cv.namedWindow(Player.WIN_NAME)
        cv.setWindowProperty(Player.WIN_NAME, cv.WND_PROP_FULLSCREEN, 1.0)
        cv.moveWindow(Player.WIN_NAME, Player.WIN_XY[0], Player.WIN_XY[1])

    def play(self, input_source: str, delay=1):

        print(f"New file is playing: {input_source}")
        self.input_source = input_source
        self.input_fs = FrameStream(input_source)
        self.delay = delay

        while True:
            frame, frame_name, frame_cnt = self.input_fs.next_frame()
            if frame is None:
                self.restart_track()
                continue

            out_frame = frame  # no processing
            self.__draw_source_name(out_frame)
            self.__draw_delay(out_frame)

            cv.imshow(Player.WIN_NAME, out_frame)
            ch = cv.waitKey(0 if self.frame_mode else self.delay)

            if WatchDog.new_file_arrived:
                self.change_track()
                continue
            if ch == ord('q'):
                return
            elif ch == ord('g'):
                self.frame_mode = False
                continue
            elif ch == ord(' '):
                self.frame_mode = True
                continue
            elif ch == ord('s'):
                snap_file_name = f'img/snap_{datetime.datetime.now().strftime("%d%m%Y_%H%M%S")}.png'
                cv.imwrite(snap_file_name, out_frame)
                continue
            elif ch == ord('z'):
                self.zone_draw_mode = not self.zone_draw_mode
            elif ch == ord('+'):
                self.delay = self.delay * 2
                continue
            elif ch == ord('-'):
                self.delay = self.delay // 2
                continue


        del self.input_fs

        # print(f"Finish. Duration={input_fs.total_time():.0f} sec, {input_fs.frame_cnt} frames,  fps={input_fs.fps():.1f} f/s")

    def restart_track(self):
        del self.input_fs
        self.input_fs = FrameStream(self.input_source)

    def change_track(self):
        if not WatchDog.new_file_arrived:
            logging.error(f"change track: ????? not arrived yet.")
            return
        new_file_name = WatchDog.get_new_file()
        if new_file_name == self.input_source:
            logging.error(f"change track: the same file name {new_file_name}")
            return
        logging.debug(f"Player.change_track:   {self.input_source}  ->  {new_file_name}")

        self.input_source = new_file_name
        del self.input_fs
        self.input_fs = FrameStream(self.input_source)

    def __draw_source_name(self, frame):
        # add name if input source to frame
        if self.input_source[:4] == "rtsp":
            name = "RTSP Stream"
        else:
            name = os.path.splitext(os.path.basename(self.input_source))[0]
        cv.putText(frame, f"{name}", (200, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def __draw_delay(self, frame):
        cv.putText(frame, f"{self.delay}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

class WatchDog:
    FOLDER_TO_WATCH = "swings/"
    FILES_TO_WATCH = ["*.avi"]
    new_file_arrived: bool = False
    __new_file_name = None

    @staticmethod
    def on_closed(event):
        WatchDog.new_file_arrived = True
        WatchDog.__new_file_name = event.src_path
        logging.debug(f"Watchdog: file {event.src_path} has been closed!")

    @staticmethod
    def set_watchdog():
        my_event_handler = PatternMatchingEventHandler(WatchDog.FILES_TO_WATCH, None, True, True)
        my_event_handler.on_closed = WatchDog.on_closed

        my_observer = Observer()
        my_observer.schedule(my_event_handler, WatchDog.FOLDER_TO_WATCH, recursive=False)

        my_observer.start()

    @staticmethod
    def get_new_file():
        if not WatchDog.new_file_arrived:
            return None
        WatchDog.new_file_arrived = False
        arrived_file_name = WatchDog.__new_file_name
        WatchDog.__new_file_name = None
        return arrived_file_name

def main():
    player = Player()
    logging.debug(f"\n\n\n\n\nPlayer-monitor started: ")

    # MONITOR_MODE = True
    WatchDog.set_watchdog()
    while not WatchDog.new_file_arrived:
        time.sleep(1)
    new_file_name = WatchDog.get_new_file()
    print(f"first file arrived: {new_file_name}")
    player.play(new_file_name, delay=150)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
