from statistics import mode
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import time
import cv2
import numpy as np
import os
# Get Face
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes
# Get Gender
def getGender(blob):
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
#    print("Gender Output : {}".format(genderPreds))
#    print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
    return genderList[genderPreds[0].argmax()]
# Get Age
def getAge(blob):
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
#    print("Age Output : {}".format(agePreds))
#    print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
    return ageList[agePreds[0].argmax()]
# Get Face Position
def getFacePosition(x1, x2, y1, y2):
    return x1, x2, y1, y2
# Get Emotion
def getEmotion(x1,x2,y1,y2,gray):
    gray_image=gray.copy()
    gray_face = gray_image[y1:y2, x1:x2]
    gray_face = cv2.resize(gray_face, (emotion_target_size))

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    return emotion_text
# Change Color
def changeColor(emotion_text):
    if emotion_text == 'angry':
        color = emotion_probability * np.asarray((255, 0, 0))
    elif emotion_text == 'sad':
        color = emotion_probability * np.asarray((0, 0, 255))
    elif emotion_text == 'happy':
        color = emotion_probability * np.asarray((255, 255, 0))
    elif emotion_text == 'surprise':
        color = emotion_probability * np.asarray((0, 255, 255))
    else:
        color = emotion_probability * np.asarray((0, 255, 0))
    color = color.astype(int)
    color = color.tolist()
    return color
# Include the caffee model
faceProto = "./data/models/face_detector/opencv_face_detector.pbtxt"
faceModel = "./data/models/face_detector/opencv_face_detector_uint8.pb"
ageProto = "./data/models/cnn_age_gender_models/age_deploy.prototxt"
ageModel = "./data/models/cnn_age_gender_models/age_net.caffemodel"
genderProto = "./data/models/cnn_age_gender_models/gender_deploy.prototxt"
genderModel = "./data/models/cnn_age_gender_models/gender_net.caffemodel"

# Include OpenCV Haar Cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

# Known People ID and Padding and Font and Color and Emotion_Probability
id = 0
padding = 20
font = cv2.FONT_HERSHEY_SIMPLEX
bounder_color=(0,200,200)
text_color=(255,255,255)
emotion_probability=0.5
# Labels
names = ['George.T', 'Wayne', 'Tim', 'Jim', 'Jessica']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
#ages =[21,22,22,21,21]
#genders=['M','M','M','M','F']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)


while True:
    hasFrame, frame =cam.read()
    frame = cv2.flip(frame, 1) # Flip vertically
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# For Recognition
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minW)),
       )
       
    for(x,y,w,h) in faces:
        x1,x2,y1,y2=getFacePosition(x,(x+w),y,y+h)
        bboxes = []
        bboxes.append([x1, y1, x2, y2])
        for bbox in bboxes:
            # Get ID and Confidence
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Get Source Face
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding,frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face,1.0,(227, 227), MODEL_MEAN_VALUES,swapRB=False)
            # Get Gender
            gender=getGender(blob)
            # Get Age
            age=getAge(blob)
            # Get Emotion
            emotion_text=getEmotion(x1,x2,y1,y2,gray)
            # Change Color
            bounder_color=changeColor(emotion_text)
            # Evaluate Confidence
            if (confidence < 100):
                name=names[id]
                confidence=":{0}%".format(round(100 - confidence))
            else:
                name="Unknown"
                confidence=":{0}%".format(round(100 - confidence))
                    
            cv2.rectangle(frame,(x1,y1),(x2,y2),bounder_color,4)
            cv2.rectangle(frame,(x1, y2 - 30),(x+w+2,y+h+100), bounder_color,cv2.FILLED)
            cv2.putText(frame,str(name)+str(confidence),(x1+5,y2-5),font, 1,text_color,2)
            cv2.putText(frame,'age:'+str(age),(x1+5,y2+25),font,1,text_color,2)
            cv2.putText(frame,'sex:'+str(gender),(x1+5,y2+55),font,1,text_color,2)
            cv2.putText(frame,emotion_text,(x1+5,y2+85),font,1,text_color,2)
            cv2.imshow('2019_SummerProject Demo',frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
