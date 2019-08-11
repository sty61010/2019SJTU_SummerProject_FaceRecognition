from statistics import mode
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import face_recognition
import cv2
import numpy as np
import time
import os
# Draw Labels
def drawLabels(left,right,top,bottom):
    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom),bounder_color, 2)
    cv2.rectangle(frame,(left, bottom-35),(right, bottom+100),bounder_color,cv2.FILLED)
    cv2.putText(frame,name+':'+str(accuracy),(left + 6, bottom - 5),font,1.0, text_color, 1)
    cv2.putText(frame,'age:'+str(age),(left+6,bottom+25),font, 1.0,text_color, 1)
    cv2.putText(frame,str(sex), (left + 6, bottom+55), font, 1.0,text_color, 1)
    cv2.putText(frame,lock, (left + 6, bottom+85), font, 1.0,text_color, 1)
    return
# Webcam
video_capture = cv2.VideoCapture(0)

# Load Picture
Jessica_image = face_recognition.load_image_file("Jessica.jpeg")
Jessica_face_encoding = face_recognition.face_encodings(Jessica_image)[0]
George_image = face_recognition.load_image_file("George.jpg")
George_face_encoding = face_recognition.face_encodings(George_image)[0]

# Labels
known_face_encodings = [Jessica_face_encoding,George_face_encoding]
known_face_names = ["Jessica.C","George.T"]
ages =[20,21]
sexs=['F','M']

# Variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
font = cv2.FONT_HERSHEY_DUPLEX
text_color=(255,255,255)
bounder_color=(0,0,255)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            print(face_distances)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                accuracy="{0}%". format(round(100-100*face_distances[best_match_index]))
                if 100-100*face_distances[best_match_index] >50:
                    name=known_face_names[best_match_index]
                    age=ages[best_match_index]
                    sex=sexs[best_match_index]
                    lock='Unlocked'
                else :
                    name='Unknown'
                    age=ages[best_match_index]
                    sex=sexs[best_match_index]
                    lock='Locked'
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw Labels
        drawLabels(left,right,top,bottom)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

