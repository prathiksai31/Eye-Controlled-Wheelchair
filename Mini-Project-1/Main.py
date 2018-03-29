# This is the code for the project "Eye Controlled Wheelchair" done for Mini Project - 1 under Electronics Engineering
# syllabus 
#@uthor1: Nikhil Pinto
#@uthor2: Sai Prathik


import numpy as np
import cv2
import time
import serial


# Serial Commands........................................................................................

# Initialize serial

port = "COM3"
baud = 9600
ser = serial.Serial(port, baud, timeout=None)

#........................................................................................................


#Creation of Window named "settings"
cv2.namedWindow('settings', flags=cv2.WINDOW_NORMAL)


# Open window with the required sliders.
def nothing():
    pass

#Creation of Trackbars........................................................................................................

cv2.createTrackbar('lower_hue', 'settings', 0, 255, nothing)
cv2.createTrackbar('lower_sat', 'settings', 0, 255, nothing)
cv2.createTrackbar('lower_val', 'settings', 0, 255, nothing)
cv2.createTrackbar('upper_hue', 'settings', 0, 255, nothing)
cv2.createTrackbar('upper_sat', 'settings', 0, 255, nothing)
cv2.createTrackbar('upper_val', 'settings', 0, 255, nothing)

# Using Haarcascades for localizing Eyes and Face........................................................................................................

face_cascade = cv2.CascadeClassifier('C:\Users\NIKHIL\Miniconda2\Library\share\OpenCV\haarcascades\\'
                                     'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Users\NIKHIL\Miniconda2\Library\share\OpenCV\haarcascades\\'
                                    'haarcascade_lefteye_2splits.xml')
#Initializing Variables........................................................................................................

no_of_frames_left = 0
no_of_frames_right = 0
no_of_frames_centre = 0
no_of_frames_stop = 0

trigger_start_centre = 0
trigger_start_left = 0
trigger_start_right = 0

counter_centre = 0
counter_right = 0
counter_left = 0

eye_closed = 0
stop_when_closed_threshold = 7

calibration_done = False
skip_calibration = False
motor_has_not_stop = True

centre_calib_values = []
left_calib_values = []
right_calib_values = []

length_centre_values = []
length_right_values = []
length_left_values = []

lower_sat = 0
lower_hue = 0
lower_val = 0
upper_sat = 0
upper_hue = 0
upper_val = 0

previous_detcted_value = 'stop'

# creating file for storing calibration values........................................................................................................

stored_calibration_centre = open('calibration_centre_text.txt', 'r+')
stored_calibration_right = open('calibration_right_text.txt', 'r+')
stored_calibration_left = open('calibration_left_text.txt', 'r+')
hsv_file_values = open('hsv_values.txt', 'r+')

# reading files to the list........................................................................................................

for line in stored_calibration_centre.read().split():  # Read centre values
    length_centre_values.append(float(line))


avg_length_centre = length_centre_values[-1]

for line in stored_calibration_right.read().split():  # Read right values
    length_right_values.append(float(line))


avg_length_right = length_right_values[-1]

for line in stored_calibration_left.read().split():  # Read left values
    length_left_values.append(float(line))

avg_length_left = length_left_values[-1]

#Creating Dictionary for the direction commands viz; Left, Center, Right and Stop

command_dict = {
    'left': '1',
    'center': '2',
    'right': '3',
    'stop': '4'
}
#......................................................................................................................................................................
detected_value = None

camera_port = 0
# noinspection PyArgumentList
cap = cv2.VideoCapture(camera_port)

#Send command serially to bot
def write_command(command):
    if ser.isOpen():
        print "Sent command: " + command
        for character in command:
            ser.write(character)
    else:
        print "Send-Serial not open!"

# -------------------------------------------------------------------#

#Exectute until eyes are closed

while True:
    start_time = time.clock()
    ret, img = cap.read()
    cv2.imshow('src_image', img)
    orig_height, orig_width, orig_channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detection of faces from the feed
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + (w / 2)]

        roi_color = img[y:y + h, x:x + (w / 2)]
        
        # Detection of Eyes from the face loop
        eyes = eye_cascade.detectMultiScale(roi_color, 1.3, 5)

  # -------------------------------------------------------------------#    
        #Executes only if calibration is done and motor is running
        if calibration_done:

            if motor_has_not_stop:

                if len(eyes) == 0:
                    eye_closed += 1
		    #If eyes are closed for 7 frames, stop the motor

                    if stop_when_closed_threshold <= eye_closed:
                        eye_closed = 0
                        detected_value = 'stop'
                        motor_has_not_stop = False
                        if detected_value in command_dict.keys():
                            write_command(command_dict[detected_value])
                            cv2.waitKey(3000)

                else:
                    eye_closed = 0
	#If motor is moving, count the number of frames for which eyes are closed
        if not motor_has_not_stop:

            if len(eyes) == 0:
                eye_closed += 1
                if stop_when_closed_threshold <= eye_closed:
                    eye_closed = 0
                    motor_has_not_stop = True

            else:
                eye_closed = 0

        for (ex, ey, ew, eh) in eyes:

            eye_mask = roi_color[ey: ey + eh, ex: ex + ew]
            
            # eye_mask is the masked portion of the detected eye extracted from roi_color
            r = 200.0 / eye_mask.shape[1]
            dim = (200, int(eye_mask.shape[0] * r))
            resized_eye = cv2.resize(eye_mask, dim, interpolation=cv2.INTER_AREA)

            hsv = cv2.cvtColor(resized_eye, cv2.COLOR_BGR2HSV)

            lower_sat = cv2.getTrackbarPos('lower_sat', 'settings')
            lower_hue = cv2.getTrackbarPos('lower_hue', 'settings')
            lower_val = cv2.getTrackbarPos('lower_val', 'settings')
            upper_sat = cv2.getTrackbarPos('upper_sat', 'settings')
            upper_hue = cv2.getTrackbarPos('upper_hue', 'settings')
            upper_val = cv2.getTrackbarPos('upper_val', 'settings')
            # -------------------------------------------------------------------#
            # define range of white color in HSV
            
            lower_white = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
            upper_white = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)

            # Threshold the HSV image to get only white colors
            mask = cv2.inRange(hsv, lower_white, upper_white)

            frame_height, frame_width, channels = hsv.shape
            mask[0:frame_height / 2, 0:frame_width] = 255

            
            cv2.rectangle(mask, (0, 0), (frame_width - 1, frame_height - 1), 255, 5)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(resized_eye, resized_eye, mask=mask)

            cv2.imshow('original mask', mask)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            dimension_of_original_image = frame_width / frame_height
            area_of_og = frame_width * frame_height

            if len(contours) > 0:
                contour_areas = []

                for i in range(0, len(contours)):
                    area = cv2.contourArea(contours[i])
                    contour_areas.append((area, i))

                contour_areas.sort()
                max_area_cnt = None
                for i in xrange(len(contour_areas) - 1, -1, -1):

                    pt_inside_rect = False

                    max_area_cnt_index = contour_areas[i][1]
                    blinked_area = contour_areas[i][0]
                    max_area_cnt = contours[max_area_cnt_index]
                    epsilon = 0.01 * cv2.arcLength(max_area_cnt, True)
                    approx = cv2.approxPolyDP(max_area_cnt, epsilon, True)

                    for points in approx:
                        point_x = points[0][0]
                        point_y = points[0][1]

                        if 3 < point_x < frame_width - 3 and 3 < point_y < frame_height - 3:
                            pt_inside_rect = True
                    if pt_inside_rect:
                        break               
                cv2.drawContours(resized_eye, [max_area_cnt], 0, (0, 255, 0), 3)               
                M = cv2.moments(max_area_cnt)
                eye_center_x = int(M["m10"] / M["m00"])
                eye_center_y = int(M["m01"] / M["m00"])

                hypo_x = eye_center_x ** 2
                hypo_y = (frame_height - eye_center_y) ** 2
                hypo_final = hypo_x + hypo_y
                length = hypo_final ** 0.5
                print 'current length', length

                if cv2.waitKey(80) & 0xFF == ord('s'):
                    skip_calibration = True
                    

                # --------calibration starts ---------------

                if cv2.waitKey(50) & 0xFF == ord('c'):
                    print 'CENTRE STARTED'
                    trigger_start_centre = 1
                    counter_centre = 30
                    stored_calibration_centre.seek(0)

                if trigger_start_centre == 1:
                    cv2.line(resized_eye, (eye_center_x, eye_center_y), (0, frame_height), (255, 0, 0), 3)
                    cv2.rectangle(img, (0, 0), (orig_width, orig_height), (255, 0, 0), 10)
                    hypo_x = eye_center_x ** 2
                    hypo_y = (frame_height - eye_center_y) ** 2
                    hypo_final = hypo_x + hypo_y
                    length = hypo_final ** 0.5
                    length_centre_values.append(length)
                    stored_calibration_centre.write('%f \t' % length)
                    centre_calib_values.append((eye_center_x, eye_center_y))
                    counter_centre -= 1
                    if counter_centre <= 1:
                        trigger_start_centre = 0
                        stored_calibration_centre.truncate()
                        avg_length_centre = sum([c for c in length_centre_values]) / float(len(length_centre_values))
                        stored_calibration_centre.write('%f \t' % avg_length_centre)
                        print 'CENTRE CALIBRATION ENDED '
                        print length_centre_values

                if counter_centre == 1:
                    if cv2.waitKey(0) & 0xFF == ord('l'):
                        print 'LEFT STARTED'
                        trigger_start_left = 1
                        counter_left = 30
                        counter_centre = 0
                        stored_calibration_left.seek(0)

                if trigger_start_left == 1:
                    cv2.line(resized_eye, (eye_center_x, eye_center_y), (0, frame_height), (0, 255, 0), 3)
                    cv2.rectangle(img, (0, 0), (orig_width, orig_height), (0, 255, 0), 10)
                    hypo_x = eye_center_x ** 2
                    hypo_y = (frame_height - eye_center_y) ** 2
                    hypo_final = hypo_x + hypo_y
                    length = hypo_final ** 0.5
                    length_left_values.append(length)
                    stored_calibration_left.write('%f \t' % length)
                    left_calib_values.append((eye_center_x, eye_center_y))
                    counter_left -= 1
                    if counter_left <= 1:
                        trigger_start_left = 0
                        stored_calibration_left.truncate()
                        avg_length_left = sum([c for c in length_left_values]) / float(len(length_left_values))
                        stored_calibration_left.write('%f \t' % avg_length_left)
                        print 'LEFT CALIBRATION ENDED '
                        print length_left_values

                if counter_left == 1:
                    if cv2.waitKey(0) & 0xFF == ord('r'):
                        print 'RIGHT STARTED'
                        trigger_start_right = 1
                        counter_right = 30
                        counter_left = 0
                        stored_calibration_right.seek(0)

                if trigger_start_right == 1:
                    cv2.line(resized_eye, (eye_center_x, eye_center_y), (0, frame_height), (0, 0, 255), 3)
                    cv2.rectangle(img, (0, 0), (orig_width, orig_height), (0, 0, 255), 10)
                    hypo_x = eye_center_x ** 2
                    hypo_y = (frame_height - eye_center_y) ** 2
                    hypo_final = hypo_x + hypo_y
                    length = hypo_final ** 0.5
                    length_right_values.append(length)
                    stored_calibration_right.write('%f \t' % length)
                    right_calib_values.append((eye_center_x, eye_center_y))
                    counter_right -= 1
                    if counter_right <= 1:
                        trigger_start_right = 0
                        stored_calibration_right.truncate()
                        avg_length_right = sum([c for c in length_right_values]) / float(len(length_right_values))
                        stored_calibration_right.write('%f \t' % avg_length_right)
                        calibration_done = True
                        print 'RIGHT CALIBRATION DONE '
                        print length_right_values

                        if cv2.waitKey(0) & 0xFF == ord('g'):
                            continue

                # --------------calibration ends---------------------------------------------------
                
                if motor_has_not_stop:

                    if calibration_done or skip_calibration:

                        detected_value = None
                        print 'average length centre ', avg_length_centre
                        
                        print 'average length left ', avg_length_left
                        
                        print 'average length right ', avg_length_right

                        if int(avg_length_right) + 1 <= int(length) <= int(avg_length_left) - 1:
                            no_of_frames_right = 0
                            no_of_frames_left = 0
                            no_of_frames_centre += 1
                            if no_of_frames_centre % 4 == 0:
                                cv2.line(resized_eye, (eye_center_x, eye_center_y), (0, frame_height),
                                         (255, 0, 0), 3)

                                detected_value = 'center'
                                print 'CENTRE DETECTED'
                                cv2.circle(resized_eye, (eye_center_x, eye_center_y), 7, (255, 0, 0), -1)
                                no_of_frames_centre = 0
                        
                        elif int(length) <= int(avg_length_right):
                            no_of_frames_centre = 0
                            no_of_frames_left = 0
                            no_of_frames_right += 1
                            if no_of_frames_right % 4 == 0:
                                cv2.line(resized_eye, (eye_center_x, eye_center_y), (0, frame_height),
                                         (0, 0, 255), 3)

                                detected_value = 'right'
                                print 'RIGHT DETECTED'
                                cv2.circle(resized_eye, (eye_center_x, eye_center_y), 7, (0, 0, 255), -1)
                                no_of_frames_right = 0

                        elif int(avg_length_left) <= int(length):
                            no_of_frames_centre = 0
                            no_of_frames_right = 0
                            no_of_frames_left += 1
                            if no_of_frames_left % 4 == 0:
                                cv2.line(resized_eye, (eye_center_x, eye_center_y), (0, frame_height),
                                         (0, 255, 0), 3)

                                detected_value = 'left'
                                print 'LEFT DETECTED'
                                cv2.circle(resized_eye, (eye_center_x, eye_center_y), 7, (0, 255, 0), -1)
                                no_of_frames_left = 0

                        else:
                            print 'NOT IN RANGE'
                            no_of_frames_right = 0
                            no_of_frames_left = 0
                            no_of_frames_centre = 0
                            cv2.circle(resized_eye, (eye_center_x, eye_center_y), 7, (255, 0, 255), -1)
                    if detected_value in command_dict.keys():
                        if not previous_detcted_value == detected_value:
                            write_command(command_dict['stop'])
                            cv2.waitKey(1000)
                        write_command(command_dict[detected_value])
                        previous_detcted_value = detected_value

                cv2.imshow('resized_eye', resized_eye)
                print ''
                print ''                             
    if cv2.waitKey(90) & 0xFF == ord('p'):
        detected_value = 'stop'
        if detected_value in command_dict.keys():
            write_command(command_dict[detected_value])
            cv2.waitKey(1000)
        # break

    k = cv2.waitKey(100) & 0xFF  # if 'Esc' pressed, exit immediately
    if k == 27:
        hsv_file_values.write('%f \t' % lower_sat)
        hsv_file_values.write('%f \t' % lower_hue)
        hsv_file_values.write('%f \t' % lower_val)
        hsv_file_values.write('%f \t' % upper_sat)
        hsv_file_values.write('%f \t' % upper_hue)
        hsv_file_values.write('%f \t' % upper_val)

        stored_calibration_centre.close()
        stored_calibration_right.close()
        stored_calibration_left.close()
        hsv_file_values.close()
        break

cap.release()
cv2.destroyAllWindows()
# ---------------------------END----------------------------------------#
