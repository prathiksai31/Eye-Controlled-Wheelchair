# This program is written to get real time values of Hsv for a given lighting condition. As lighting changes, the HSV
# values also change. Using a slider, we can get the values of a particular color and write those in our required file.

import cv2
import numpy as np

port = 1  # 1 is used for webcam, default 0
#  noinspection PyArgumentList
cap = cv2.VideoCapture(port)

# Creating a window for later use
cv2.namedWindow('settings', flags=cv2.WINDOW_NORMAL)

# Starting with 100's to prevent error while masking


def nothing():
    pass

# Creating track bar


cv2.createTrackbar('lower_hue', 'settings', 0, 255, nothing)
cv2.createTrackbar('lower_sat', 'settings', 0, 255, nothing)
cv2.createTrackbar('lower_val', 'settings', 0, 255, nothing)
cv2.createTrackbar('upper_hue', 'settings', 0, 255, nothing)
cv2.createTrackbar('upper_sat', 'settings', 0, 255, nothing)
cv2.createTrackbar('upper_val', 'settings', 0, 255, nothing)

while True:

    ret, img = cap.read()
    # Filtering Noise
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.bilateralFilter(img, 9, 75, 75)
    # converting to HSV
    cv2.imshow('blur', blur)
    hsv = cv2.cvtColor(edges, cv2.COLOR_BGR2HSV)

    lower_sat = cv2.getTrackbarPos('lower_sat', 'settings')
    lower_hue = cv2.getTrackbarPos('lower_hue', 'settings')
    lower_val = cv2.getTrackbarPos('lower_val', 'settings')
    upper_sat = cv2.getTrackbarPos('upper_sat', 'settings')
    upper_hue = cv2.getTrackbarPos('upper_hue', 'settings')
    upper_val = cv2.getTrackbarPos('upper_val', 'settings')

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
    upper_white = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow('original mask', mask)

    # Get contours of image
    image, contours, heirachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    # If contours detected
    if len(contours) > 0:
        contour_areas = []

        # Store each Index value and Area of contour
        for i in range(0, len(contours)):
            area = cv2.contourArea(contours[i])
            contour_areas.append((area, i))

        # Sort areas in ascending order
        contour_areas.sort()
        max_area_cnt = None
        print 'contour area : ', contour_areas[-1]
        # Get index of largest area contour
        max_area_cnt_index = contour_areas[-1][1]
        max_area_cnt = contours[max_area_cnt_index]

        # Draw contour on image
        cv2.drawContours(img, [max_area_cnt], 0, (0, 255, 0), 3)

    cv2.imshow('original', img)

    # Press 'Esc" to terminate Program
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
