# player_processed (use frame_processor from external file)
# modes: frame-by-frame, write, markup
# keys:
#   ' ' - frame mode, 'g' - go (stream),
#   's' - shot
#   'z' - on/off zone  drawing
#   left-mouse - add corner to zone, right-mose - reset zone
#   '1'-'9' - delays


import datetime
import cv2 as cv

from util import FrameStream
from zone import OnePointZone

# inp_source_name = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
inp_source_name = 'video/nb-profil-1 (daylight).avi' # 0.avi b2_cut phone-profil-evening-1.mp4 fac-2 nb-profil-1 (daylight) black3 - ne vidno kuda letit
# inp_source_name = '/run/user/1000/gvfs/mtp:host=Xiaomi_Redmi_Note_8_Pro_fukvv87l8pbuo7eq/Internal shared storage/DCIM/Camera/tst2.mp4'

out_file_name = 'video/out2.avi'

frame_mode_initial = False
# frame_mode_initial = True

write_mode = True if inp_source_name[0:4] == 'rtsp' else False
# write_mode = False

delay_initial = 1
delay_multiplier = 60
inp_frame_shape = (1280, 720)  # (1920, 1080)
need_frame_processor = True  # False

frame_proc_fname = None  # no proc
end_stream_proc = None
if need_frame_processor:
    try:
        from cutter_frame_proc import frame_processor, end_stream_processor # zone_frame_proc
        frame_proc = frame_processor
        end_stream_proc = end_stream_processor
    except ImportError as error:
        print("There is no file 'frame_processor.py' so no processors will be used")


def main():
    frame_mode = frame_mode_initial

    zone_draw_mode = True  # True - draw active zone (corners_lst) on all images
    cv.namedWindow('out_frame')
    OnePointZone.zone_init('out_frame', need_load=False)

    delay = delay_initial
    fs = FrameStream(inp_source_name)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = None

    while True:
        frame, frame_name, frame_cnt = fs.next_frame()
        if frame is None:
            break
        out_frame = frame_proc(frame, frame_cnt) if frame_proc else frame
        cv.putText(out_frame, f"D:{delay / delay_multiplier:.0f}/{frame_cnt}", (5, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        OnePointZone.new_frame(out_frame)
        if zone_draw_mode:
            OnePointZone.draw_zone(out_frame)

        cv.imshow(f'out_frame', out_frame)

        if write_mode:
            if not out:
                out_shape = (out_frame.shape[1], out_frame.shape[0])
                out = cv.VideoWriter(out_file_name, fourcc, 20.0, out_shape)
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
        elif ch == ord('z'):
            zone_draw_mode = not zone_draw_mode

    OnePointZone.zone_save()
    if end_stream_proc:
        end_stream_proc(frame_cnt)

    print(
        f"Finish. Duration={fs.total_time():.0f} sec, {fs.frame_cnt} frames,  fps={fs.fps():.1f} f/s")

    del fs
    if write_mode:
        out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
