import re
import cv2 as cv

from start_zone import StartZone, StartBall

SWING_CLIP_PREFIX = "video/swings/"
NEED_TRANSPOSE = True
NEED_FLIP = True
INPUT_SCALE = 0.7


class History:
    states_string: str = ""

    @classmethod
    def save_state(cls, state: str, frame):
        # status_history += status
        # frames_buffer.append(frame.copy())
        # logging.debug(f"{len(frames_buffer)=}")
        pass

    @classmethod
    def write_swing(cls, r):
        pass

    @classmethod
    def reset(cls):
        # status_history = ''
        # frames_buffer.clear()
        pass


class FrameProcessor:
    def __init__(self, filename=""):
        self.filename = filename
        print("swing cutter proc")

    def process_frame(self, frame, frame_cnt):

        if frame_cnt == 1:
            pass  # OnePointZone.reset_zone_point()  # it points to ball so we have to re-init it each time

        if INPUT_SCALE != 1.0:
            cv.resize(frame, None, fx=INPUT_SCALE, fy=INPUT_SCALE)  # !!!
        if NEED_TRANSPOSE:
            frame = cv.transpose(frame)
        if NEED_FLIP:
            frame = cv.flip(frame, 1)

        if not StartZone.find(frame):
            return frame

        start_zone_state = StartZone.current_state(frame)
        History.save_state(start_zone_state, frame)
        frame = StartZone.draw(frame)

        r = re.search('B{7}B*[MB]{0,7}E{15}$', History.states_string)  # B{7}[MB]*E{7}$
        if r:
            History.write_swing(r)
            History.reset()

    def end_stream(self, frame_cnt):
        pass
