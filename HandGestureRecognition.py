
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import imutils
from sklearn.metrics import pairwise
 
import webbrowser



def segment(image, threshold=20):
    global back_ground
    diff = cv2.absdiff(back_ground.astype("uint8"), image)

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    (contours, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

def run_avg(image, aWeight):
    global back_ground
    if back_ground is None:
        back_ground = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, back_ground, aWeight)

def count(thresholded, segmented):
    chull = cv2.convexHull(segmented)

    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    radius = int(0.75 * maximum_distance)

    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    (contours, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return min(count, 5), cX, cY, radius


flag = 0

aWeight = 0.5
cap = cv2.VideoCapture(0)

top, right, bottom, left = 0, 650, 500, 1000

number_of_frames = 0
back_ground = None
while True:
    try:
        check, frame = cap.read()
        frame = imutils.resize(frame, width=1000)

        frame = cv2.flip(frame, 1)

        frame_clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
                               


        if number_of_frames < 39:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand

                cv2.drawContours(frame_clone, [segmented + (right, top)], -1, (110, 2, 76))

                fingers, cX, cY, radius = count(thresholded, segmented)

                cv2.putText(frame_clone, str(fingers), (700, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (110, 2, 76), 5)
                
                cv2.circle(thresholded, (cX, cY), radius, 255, 1)

                cv2.imshow("Thesholded", thresholded)

                if fingers == 5:
                     img = cv2.imread("/home/msd/Desktop/msd.jpg")
                     cv2.namedWindow("Image")
                     cv2.resizeWindow("Image", 1000, 1000)
                     cv2.imshow("Image", img)

                elif fingers == 1:
                     cv2.destroyWindow("Image")
                       
                       
                elif fingers == 2:
                     webbrowser.open('http://google.com', new=2)
                       
                elif fingers == 3:
                       flag = 1
                       break
                       
 

        cv2.rectangle(frame_clone, (left, top), (right, bottom), (245, 87, 66), 10)

        number_of_frames += 1
        if number_of_frames == 40:
            print("Back Ground is Recognized, Ready!")
        cv2.namedWindow("Window")
        cv2.resizeWindow("Window", 1000, 1000)
        cv2.imshow("Window", frame_clone)
        
        key = cv2.waitKey(1)

        if key == ord('r'):
            num_frames = 0
            back_ground = None
        elif key == ord('q'):
            break
    except:
        pass

cap.release()
cv2.destroyAllWindows()

if flag == 1:


    def nothing(x):
        pass



    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LH","Tracking", 0, 255, nothing)
    cv2.createTrackbar("LS","Tracking", 0, 255, nothing)
    cv2.createTrackbar("LV","Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH","Tracking", 255, 255, nothing)
    cv2.createTrackbar("US","Tracking", 255, 255, nothing)
    cv2.createTrackbar("UV","Tracking", 255, 255, nothing)
    

    while True:
        frame = cv2.imread('/home/msd/smarties.png')
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        l_h =cv2.getTrackbarPos("LH", "Tracking")
        l_s =cv2.getTrackbarPos("LS", "Tracking")    
        l_v =cv2.getTrackbarPos("LV", "Tracking")    
    
    
        u_h =cv2.getTrackbarPos("UH", "Tracking")
        u_s =cv2.getTrackbarPos("US", "Tracking")
        u_v =cv2.getTrackbarPos("UV", "Tracking")

    
        l_b = np.array([l_h, l_s , l_v])
        u_b = np.array([u_h, u_s, u_v])
    
        mask = cv2.inRange(hsv, l_b, u_b)
        res = cv2.bitwise_and(frame, frame, mask )
        
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)
    
        key = cv2.waitKey(1)
        if key ==27 :
            break
    

    cv2.destroyAllWindows()

