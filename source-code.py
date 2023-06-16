# import the necessary packages
import csv
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sendgrid
import os
from sendgrid.helpers.mail import *

# initialize array & counter
b_px = []
w_px = []
b_perc = []
w_perc = []
tot_px = []
a = []
counter = 1500


# initialize the camera and grab a reference to the raw camera capture
#544, 480
#640, 480
#720, 480
camera = PiCamera()
camera.resolution = (720, 480)
camera.framerate = 30
camera.rotation = 0
rawCapture = PiRGBArray(camera, size = (720, 480))
firstFrame = None

# allow the camera to adjust to lighting/white balance
time.sleep(2)

# initiate video or frame capture sequence
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw array representation of the image
    frame = f.array
    
    # convert imags to grayscale &  blur the result
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # inittialize firstFrame which will be used as a reference
    if firstFrame is None:
        firstFrame = gray
        rawCapture.truncate(0)
        continue
    
    # obtain difference between frames
    frameDelta = cv2.absdiff(gray, firstFrame)

    # coonvert the difference into binary & dilate the result to fill in small holes
    thresh = cv2.threshold(frameDelta, 20, 555, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
      
    # show the result
    cv2.imshow("Delta + Thresh", thresh)


    # find contours or continuous white blobs in the image
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # find the index of the largest contour
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        
        
    for c in contours:
        if len(contours) > 5:
            for a,b,c,d,e in (contours[i:i+5] for i in range(len(contours)-5)):
                if e.all() > b.all():
                    print("Motion Detected!")
                    # utilizing sendgrid api for motion detection 
                    sg = sendgrid.SendGridAPIClient(api_key='API KEY HERE')
                    from_email = Email("SENDER EMAIL")
                    to_email = To("RECIPIENT EMAIL")
                    subject = "Motion Detection" # email subject
                    content = Content("text/plain", "\nALERT!!\nMotion Detected!") #email content
                    mail = Mail(from_email, to_email, subject, content)
                    response = sg.client.mail.send.post(request_body=mail.get())
                    print(response.status_code)
                    print(response.body)
                    print(response.headers)


        # draw a bounding box/rectangle around the larget contour
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        area = cv2.contourArea(cnt)
        
        # print area to the terminal
        print("Area:", area)
        
        
        # counting the number of pixels
        white_px = np.sum(thresh == 255)
        black_px = np.sum(thresh == 0)
        total_px = white_px+black_px
        # % of white & black pixels
        percentage_white = round((white_px/total_px)*100,2)
        percentage_black = round((black_px/total_px)*100,2)
        # print values to terminal
        while counter>0:
            #print('Number of white pixels:',white_px) 
            #print('% of white pixels detected:', percentage_white)
            #print('Number of black pixels:', black_px)
            #print('% of black pixels detected:', percentage_black)
            counter-=1
            break
        
        # add text to the frame
        cv2.putText(frame, str(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # append values to lists
        while counter>0:
            b_px.append(black_px)
            w_px.append(white_px)
            b_perc.append(percentage_black)
            w_perc.append(percentage_white)
            tot_px.append(total_px)
            #a.append(area)
            print(counter)
            counter-=1
            break
              
        dict = {'black pixels':b_px, 'white pixels':w_px, 'black pixel %':b_perc, 'white pixel %':w_perc, 'area':area, 'total pixels':tot_px}
        df = pd.DataFrame(dict)
        df.to_csv('20_455_values.csv')
    # show the frame
    cv2.imshow("Video", frame)   

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the 'q' key is pressed then break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()