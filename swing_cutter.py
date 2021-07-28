
import cv2 as cv

SWING_CLIP_PREFIX = "video/swings/"
NEED_TRANSPOSE = True
NEED_FLIP = True
INPUT_SCALE = 0.7

class FrameProcessor:
    def __init__(self):
        pass

    def process_frame(self, frame, frame_cnt, file_name=""):
        if frame_cnt == 1:
            pass  # OnePointZone.reset_zone_point()  # it points to ball so we have to re-init it each time

        if INPUT_SCALE != 1.0:
            cv.resize(frame, None, fx=INPUT_SCALE, fy=INPUT_SCALE)  # !!!
        if NEED_TRANSPOSE:
            frame = cv.transpose(frame)
        if NEED_FLIP:
            frame = cv.flip(frame, 1)



    def end_stream(self):
        pass


def frame_processor(frame, frame_cnt):


    get_start_area(frame)
    if StartArea.x is None:
        return frame

    status = get_start_area_status(frame)
    status_history += status
    frames_buffer.append(frame.copy())
    logging.debug(f"{len(frames_buffer)=}")

    cv.rectangle(frame, (StartArea.x, StartArea.y), (StartArea.x + StartArea.w, StartArea.y + StartArea.h), (255, 0, 0), 1)
    # cv.drawContours(frame, [StartArea.contour], 0, (0, 0, 255), 3)
    cv.putText(frame, f"{status}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    r = re.search('B{7}B*[MB]{0,7}E{15}$', status_history)  # B{7}[MB]*E{7}$
    logging.debug(f"{frame_cnt=}:  {status=}, {status_history=}")
    if r:
        out_file_name = write_swing_clip(r)
        print(f"hit!!!!  {frame_cnt=} {out_file_name=}")
        logging.debug(f"Hit: {r.string=}  {status_history=} {r.span()=}")
        status_history = ''
        frames_buffer.clear()
