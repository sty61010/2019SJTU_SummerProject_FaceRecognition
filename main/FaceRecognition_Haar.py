''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['George.T', 'Wayne', 'Tim', 'Jim', 'Jessica']
ages =[21,22,22,21,21]
sexs=['M','M','M','M','F']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,200,200), 4)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
#        print(id)
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
#            print(id)
            name = names[id]
#            print(id)
            confidence = "  {0}%".format(round(100 - confidence))
            age=ages[id]
            sex=sexs[id]
        else:
            names = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            age=ages[id]
            sex=sexs[id]
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(img, (x, y+h - 35), (x+w+2,y+h+60), (0, 200, 200), cv2.FILLED)
        cv2.putText(img, str(name)+str(confidence), (x+5,y+h-5), font, 1, (255,255,255), 2)
        cv2.putText(img, 'age:'+str(age), (x+5,y+h+25), font, 1, (255,255,255), 2)
        cv2.putText(img, 'sex:'+str(sex), (x+5,y+h+55), font, 1, (255,255,255), 2)

    cv2.imshow('camera',img) 

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
