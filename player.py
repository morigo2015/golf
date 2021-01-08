# player
# modes: frame-by-frame, write, markup
# keys: g,q,space,s, ->, PgDn

import datetime
import logging
import cv2 as cv

from util import FrameStream


class Player:
    # default values:
    inp_source_name = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
    out_file_name = 'video/player-out.avi'
    write_mode = True  # if inp_source_name[0:4] == 'rtsp' else False
    frame_mode_initial = False
    stamp_frame_cnt = True
    show_out_frame = True
    frame_processor = None
    end_stream_processor = None

    delay_initial = 1
    delay_multiplier = 60

    @classmethod
    def player(cls):

        logging.info(
            f"Player started. {cls.inp_source_name=}, {cls.write_mode=}, {cls.out_file_name}. "
            f"Frame processor {'is not set' if cls.frame_processor is None else 'is set'}")

        frame_mode = cls.frame_mode_initial
        frames_left = -1
        delay = cls.delay_initial
        fs = FrameStream(cls.inp_source_name)

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(cls.out_file_name, fourcc, 20.0, (1920, 1080)) if cls.write_mode else None

        while True:
            frame, frame_name, frame_cnt = fs.next_frame()
            if frame is None:
                break

            if cls.frame_processor:
                out_frame = cls.frame_processor(frame, frame_cnt, frame_name)
            else:
                out_frame = frame

            if cls.stamp_frame_cnt:
                cv.putText(out_frame, f"{frame_cnt}", (5, 12),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if cls.show_out_frame:
                cv.imshow(f'out_frame for {cls.inp_source_name}', out_frame)

            if cls.write_mode:
                out.write(out_frame)

            if frames_left == 0:
                frame_mode = True
            else:
                frames_left -= 1

            ch = cv.waitKey(0 if frame_mode else delay) & 0xFF

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
                delay = cls.delay_multiplier * (ch - ord('0'))
                frame_mode = False
                print(f"{delay=}")
            elif ch == 83:  # right arrow
                frame_mode = False
                frames_left = 10
            elif ch == 86:  # page down                logging.debug(f"right arrow")
                frame_mode = False
                frames_left = 60
            else:
                pass  # skip all unknown keys

        if cls.end_stream_processor:
            cls.end_stream_processor(frame_cnt)

        logging.info(fs.stat_string())

        del fs
        if cls.write_mode:
            out.release()
        cv.destroyAllWindows()

    # ---------------------------------------------------------------------------------------------------------------


def main():
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/debug.log", mode='w'),
            logging.StreamHandler()
        ]
    )

    Player.player()


if __name__ == '__main__':
    main()

    # inp_source_name = 'video/out2.avi'  # video/phone-2(60).mp4
    # inp_source_name = '/run/user/1000/gvfs/mtp:host=Xiaomi_Redmi_Note_8_Pro_fukvv87l8pbuo7eq/Internal shared storage/DCIM/Camera/tst2.mp4'
