import time
import picamera
from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

#HOW TO INSTALL matplotlib on your raspberry pi
#sudo apt-get install python-matplotlib

with picamera.PiCamera() as camera:

    #set camera resolutin
    camera.resolution = (512, 512)
    #flip captured image
    #camera.vflip = True
    #rotation
    camera.rotation = 90
        
    #camera.start_preview()
    time.sleep(0.1)
    timestr = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    fn = timestr + '.jpg'
    
    while True:
        #caputure
        camera.capture(fn)

        #load image
        img = cv2.imread(fn, 0)

        #grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #grayimg_list = np.array(grayimg)

        #front face detector
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        facerect = face_cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 2, minSize = (1,1))

        #eye detector
        eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
        eyerect = eye_cascade.detectMultiScale(img)
        
        #IF face is detected
        if len(facerect) > 0:
            print('Face(s) is(are) detected! : %d' % len(facerect))
            for rect in facerect:
                cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness = 3)
            plt.imshow(img)

        if len(eyerect) > 0:
            print('eye(s) is(are) detected! : %d' % len(eyerect))
            for rect in eyerect:
                cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 255, 0), thickness = 3)
                #show image
                plt.imshow(img)
      
        #TIPS : Use plt.pause(1) for realtime drawing, not plt.show()
        #plt.show()
        plt.pause(0.1)
        #cv2.imshow('camera', img)
        #cv2.imwrite(fn, img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
