import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image


def preprocess_image(img):
    bilateral_filtered_image = cv2.bilateralFilter(img, 5, 175, 175)
    img_hsv = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2HSV)

    lower_color = np.array([25, 157, 8])
    upper_color = np.array([48, 255, 48])
    mask = cv2.inRange(img_hsv, lower_color, upper_color)
   
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((7, 7), np.uint8))
    applied_mask = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    cv2.imshow("mask", mask)

    res = applied_mask
    return res

#
# def merge_contours(cnts, img):
#     """Merge these contours together. And create an image"""
#     for c in cnts:
#         hull = cv2.convexHull(c)
#         cv2.fillConvexPoly(img, hull, 0)
#     return img
#

def process_video(filename, show_output=False):
    frame_count = 0
    start = time.time()
    cap = cv2.VideoCapture(filename)
    while (True):
        ret, raw_image = cap.read()
        if ret:
            raw_image =cv2.flip(raw_image,0)
            img_cropped = raw_image
            preproc_res = preprocess_image(img_cropped)

            frame_count +=1
            if frame_count == 100:
                cv2.imwrite("test.png",img_cropped)
            if show_output:
                cv2.imshow('crop', img_cropped)
                cv2.imshow('Original Image', raw_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            end = time.time()

            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # print(images[0].shape)
            # out = cv2.VideoWriter('test_1024_720_30fps.avi', fourcc, 30.0, (674, 300))
            # for im in images:
            #     out.write(im)
            #
            # # Time elapsed
            # seconds = end - start
            # print("Time taken : {0} seconds".format(seconds))
            #
            # # save fame sequence to gif
            # # images[0].save('output/result.gif',
            # #                save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
            # # Calculate frames per second
            # fps = frame_count / seconds
            # print("Estimated frames per second : {0}".format(fps))
            # out.release()
            # plot_result(l_radius_list, r_radius_list, frame_count)
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # return l_radius_list, r_radius_list


def main():
    video_path = "/home/morylov/PycharmProjects/pupil_recognition/3/video/out2-2colors.avi"
    # video_path = "/home/morylov/PycharmProjects/pupil_recognition/3/video/out2-black.avi"
    process_video(video_path, show_output=True)



if __name__ == "__main__":
    main()
