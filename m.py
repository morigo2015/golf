# m.py
# обнаруживает пявление мяча в дальней (верхней) части кадра - как признак удара
#
import time
import numpy as np
import cv2 as cv
# video_file_name = 'long test.mp4'
# video_file_name = 'tst.mp4'
video_file_name = 'mid.mp4'

frame_mode_initial = False
nowait_mode_initial = False


def main():
    cap = cv.VideoCapture(video_file_name)  # (0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    frame_in_cnt = 0
    frame_out_cnt = 0
    frame_mode = frame_mode_initial
    nowait_mode = nowait_mode_initial

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (500, 1080))
    rej = cv.VideoWriter('rejected.avi', fourcc, 20.0, (500, 1080))
    noise = cv.VideoWriter('noise.avi', fourcc, 20.0, (500, 1080))

    backSub = cv.createBackgroundSubtractorMOG2()
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_in_cnt = frame_in_cnt + 1

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        w, h, c = frame.shape
        top = frame[0:w, 0:500]
        top_grey = cv.cvtColor(top, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(top_grey, (5, 5), 0)
        _, top_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        fgMask = backSub.apply(top_thresh)

        contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt_desc = [(idx, cnt, analyze(cnt)) for idx, cnt in enumerate(contours)]

        for idx, cnt, cnt_type in cnt_desc:
            if cnt_type == 'ball':
                frame_out_cnt = frame_out_cnt + 1
                top_copy = top.copy()
                cv.drawContours(top_copy, contours, idx, (0, 255, 0), 2)
                cv.putText(top_copy, f"frame_in: {frame_in_cnt}", (25, 1080 - 6 * 25), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 255, 255), 2)
                cv.putText(top_copy, f"frame_out: {frame_out_cnt}", (25, 1080 - 4 * 25), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 255, 255), 2)
                out.write(top_copy)
                print(f"{frame_out_cnt = }, {frame_in_cnt = }, area:{cv.contourArea(cnt)}")
            elif cnt_type == 'noise':
                noise_top = top.copy()
                cv.drawContours(noise_top, contours, idx, (0, 0, 255), 2)
                cv.putText(noise_top, f"frame_in: {frame_in_cnt}", (25, 1080 - 6 * 25), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 255, 255), 2)
                cv.putText(noise_top, f"frame_out: {frame_out_cnt}", (25, 1080 - 4 * 25), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 255, 255), 2)
                noise.write(noise_top)
            else:  # rejected frame
                rej_top = top.copy()
                cv.drawContours(rej_top, contours, idx, (255, 0, 0), 2)
                cv.putText(rej_top, f"in: {frame_in_cnt}", (25, 1080 - 6 * 25), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 255, 255), 2)
                cv.putText(rej_top, f"out: {frame_out_cnt}", (25, 1080 - 4 * 25), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 255, 255), 2)
                cv.putText(rej_top, f"type: {cnt_type}", (25, 1080 - 8 * 25), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 255, 255), 2)
                rej.write(rej_top)

        if nowait_mode:
            continue

        cv.imshow('frame', frame)  # gray
        cv.putText(top, f"in={frame_in_cnt}, out={frame_out_cnt}", (25, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.imshow('top', top)
        cv.imshow('mask', fgMask)

        ch = cv.waitKey(0 if frame_mode else 1)
        if ch == ord('q'):
            break
        elif ch == ord('g'):
            frame_mode = False
            continue
        elif ch == ord(' '):
            frame_mode = True
            continue
        elif ch == ord('n'):
            nowait_mode = True
            print("No wait mode is ON!!")

    end_time = time.time()
    print(f"Finish. Duration={end_time-start_time} sec, {frame_in_cnt=} {frame_out_cnt=} fps={frame_in_cnt/(end_time-start_time)}")
    cap.release()
    out.release()
    cv.destroyAllWindows()


def analyze(contour):
    if cv.contourArea(contour) < 1000:
        return f'noise'
    if cv.contourArea(contour) > 10000:
        return f'obstacle. area={cv.contourArea(contour)}'

    rect = cv.minAreaRect(contour)
    rect_cx = rect[0][0]
    rect_cy = rect[0][1]
    rect_w = rect[1][0]
    rect_h = rect[1][1]
    rect_ang = rect[2]

    if rect_h / rect_w > 2 or rect_h / rect_w < 0.5:
        return f'long h,w={rect_h:.0f},{rect_w:.0f}, r={rect_h / rect_w:.2f}'
    if rect_cy < 300 or rect_cy > 700:
        return f'lane cy={rect_cy:.0f}'

    perimeter = cv.arcLength(contour, True)
    hull = cv.convexHull(contour)
    hull_perimeter = cv.arcLength(hull, True)
    if perimeter / hull_perimeter > 1.1:
        print(
            f"noconv r={perimeter / hull_perimeter:.2f}. cx,cy,w,h:{rect_cx:.0f},{rect_cy:.0f},{rect_w:.0f},{rect_h:.0f}")
        return f'nonconv r={perimeter / hull_perimeter:.2f}'

    return 'ball'


if __name__ == '__main__':
    main()
