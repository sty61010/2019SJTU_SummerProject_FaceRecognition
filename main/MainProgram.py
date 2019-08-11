from statistics import mode
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from time import sleep
from PIL import Image
from imutils import face_utils, resize
from dlib import get_frontal_face_detector, shape_predictor
import face_recognition
import cv2
import numpy as np
import time


class MainProgram(object):
    # Initialize
    def __init__(self, saved=False):
        # Mode
        self.mode=1
        # Save Image
        self.saved = saved
        # Set Animation
        self.set_Animation()
        # Set Encoding Picture
        self.set_Encoding_Picture()
        # Set Labels
        self.set_Labels()
        # Set Outcomes
        self.set_Outcomes()
        # Set Variables
        self.set_Variables()
        # Set Face with OpenCV Haar Cascade
        self.set_Face_OpenCV()
        # Set Emotion
        self.set_Emotion()
        # Include Caffee Model
        self.include_Caffee()
        # Load Network
        self.load_Network()
    # Set Animation
    def set_Animation(self):
        self.listener = True  # Control
        self.video_capture = cv2.VideoCapture(0)  # Camera
        self.doing = False  # Mask
        self.speed = 0.1  # Mask Speed
        self.detector = get_frontal_face_detector()  # Dlib Detection
        self.predictor = shape_predictor("shape_predictor_68_face_landmarks.dat")  # Predictor
        self.fps = 4  # FPS
        self.animation_time = 0  # Animation time
        self.duration = self.fps * 4  # Duration
        self.fixed_time = 4  # Fixed time
        self.max_width = 600  # Size
        self.deal, self.text, self.cigarette = None, None, None  # Masks
        return
    # Set Encoding Picture
    def set_Encoding_Picture(self):
        self.Jessica_image = face_recognition.load_image_file("Jessica.jpeg")
        self.Jessica_face_encoding = face_recognition.face_encodings(self.Jessica_image)[0]
        self.George_image = face_recognition.load_image_file("George.jpg")
        self.George_face_encoding = face_recognition.face_encodings(self.George_image)[0]
        return
    # Set Labels
    def set_Labels(self):
        self.known_face_encodings = [self.Jessica_face_encoding,self.George_face_encoding]
        self.known_face_names = ["Jessica.C","George.T"]
        self.ages =[20,21]
        self.sexs=['F','M']
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']
        return
    # Set Variables
    def set_Variables(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.text_color=(255,255,255)
        self.bounder_color=(0,0,255)
        self.count=0;
        return
    # Set Outcomes
    def set_Outcomes(self):
        self.name='Unknown'
        self.age=20
        self.sex='M'
        self.lock='Locked'
        self.emotion_text='happy'
        self.accuracy="{0}%". format(round(100))
        return
    # Set Face with OpenCV Haar Cascade
    def set_Face_OpenCV(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer/trainer.yml')
        self.cascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath);
        return
    # Set Emotion
    def set_Emotion(self):
        self.emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        self.emotion_labels = get_labels('fer2013')
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.emotion_offsets = (20, 40)
        # Faces
        self.detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
        self.face_detection = load_detection_model(self.detection_model_path)
        return
    # Load Network
    def load_Network(self):
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)
        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)
        return
    # Include Caffee Model
    def include_Caffee(self):
        self.faceProto = "./data/models/face_detector/opencv_face_detector.pbtxt"
        self.faceModel = "./data/models/face_detector/opencv_face_detector_uint8.pb"
        self.ageProto = "./data/models/cnn_age_gender_models/age_deploy.prototxt"
        self.ageModel = "./data/models/cnn_age_gender_models/age_net.caffemodel"
        self.genderProto = "./data/models/cnn_age_gender_models/gender_deploy.prototxt"
        self.genderModel = "./data/models/cnn_age_gender_models/gender_net.caffemodel"
        return
    # Read Data
    def read_Data(self):
        _, data = self.video_capture.read()
        return data
    # Save Data
    def save_Data(self, draw_img):
        if not self.saved:
            return
        draw_img.save("images/%05d.png" % self.animation_time)
    # Initialize Masks
    def init_Mask(self):
        self.Console("Loading Mask...")
        self.deal, self.text, self.cigarette = (
            Image.open(x) for x in ["images/deals.png", "images/text.png", "images/cigarette.png"]
        )
    # Glasses Infomation
    def get_Glasses_Info(self, face_shape, face_width):
        left_eye = face_shape[36:42]
        right_eye = face_shape[42:48]

        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")

        y = left_eye_center[1] - right_eye_center[1]
        x = left_eye_center[0] - right_eye_center[0]
        eye_angle = np.rad2deg(np.arctan2(y, x))

        deal = self.deal.resize(
            (face_width, int(face_width * self.deal.size[1] / self.deal.size[0])),
            resample=Image.LANCZOS)

        deal = deal.rotate(eye_angle, expand=True)
        deal = deal.transpose(Image.FLIP_TOP_BOTTOM)

        left_eye_x = left_eye[0, 0] - face_width // 4
        left_eye_y = left_eye[0, 1] - face_width // 6

        return {"image": deal, "pos": (left_eye_x, left_eye_y)}
    # Cigarette Information
    def get_Cigarette_Info(self, face_shape, face_width):
        mouth = face_shape[49:68]
        mouth_center = mouth.mean(axis=0).astype("int")
        cigarette = self.cigarette.resize(
            (face_width, int(face_width * self.cigarette.size[1] / self.cigarette.size[0])),
            resample=Image.LANCZOS)
        x = mouth[0, 0] - face_width + int(16 * face_width / self.cigarette.size[0])
        y = mouth_center[1]
        return {"image": cigarette, "pos": (x, y)}
    # Get Orientation
    def get_Orientation(self, rects, img_gray):
        faces = []
        for rect in rects:
            face = {}
            face_shades_width = rect.right() - rect.left()
            predictor_shape = self.predictor(img_gray, rect)
            face_shape = face_utils.shape_to_np(predictor_shape)
            face['cigarette'] = self.get_Cigarette_Info(face_shape, face_shades_width)
            face['glasses'] = self.get_Glasses_Info(face_shape, face_shades_width)
            faces.append(face)
        return faces
    # Get Gender
    def get_Gender(self,blob):
        self.genderNet.setInput(blob)
        self.genderPreds = self.genderNet.forward()
        gender = self.genderList[self.genderPreds[0].argmax()]
#        print("Gender Output : {}".format(genderPreds))
        print("Gender : {}, conf = {:.3f}".format(gender, self.genderPreds[0].max()))
        return self.genderList[self.genderPreds[0].argmax()]
    # Get Age
    def get_Age(self,blob):
        self.ageNet.setInput(blob)
        self.agePreds = self.ageNet.forward()
        age = self.ageList[self.agePreds[0].argmax()]
#        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, self.agePreds[0].max()))
        return self.ageList[self.agePreds[0].argmax()]
    # Get Emotion
    def get_Emotion(self,x1,x2,y1,y2,gray):
        gray_image=gray.copy()
        gray_face = gray_image[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, (self.emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        self.emotion_prediction = self.emotion_classifier.predict(gray_face)
        self.emotion_probability = np.max(self.emotion_prediction)
        self.emotion_label_arg = np.argmax(self.emotion_prediction)
        emotion_text = self.emotion_labels[self.emotion_label_arg]
        print(emotion_text)
        return emotion_text
    # Get Face Position
    def getFacePosition(self,x1, x2, y1, y2):
        return x1, x2, y1, y2
    
    # Detect Faces
    def detect_Faces(self,frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                self.accuracy="{0}%". format(round(100-100*face_distances[best_match_index]))
                if (100-100*face_distances[best_match_index]) >50:
                    self.name=self.known_face_names[best_match_index]
                    self.age=self.ages[best_match_index]
                    self.sex=self.sexs[best_match_index]
                    self.lock='Unlocked'
                else :
                    self.name='Unknown'
                    self.age=self.ages[best_match_index]
                    self.sex=self.sexs[best_match_index]
                    self.lock='Locked'
            self.face_names.append(self.name)
        self.process_this_frame = not self.process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw Labels
            self.draw_Labels(left,right,top,bottom,frame)
            # Show Result
#            cv2.imshow("2019SJTU SummerProject", frame)
        return
    # Get Detections
    def get_Detections(self,frame):
#        frame = resize(frame, width=self.max_width)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(100), int(100)),)
        padding=20
        for(x,y,w,h) in faces:
            x1,x2,y1,y2=self.getFacePosition(x,(x+w),y,y+h)
            bboxes = []
            bboxes.append([x1, y1, x2, y2])
            for bbox in bboxes:
                # Get ID and Confidence
#                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
#                print(100-confidence)
                # Get Source Face
                face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding,frame.shape[1]-1)]
                blob = cv2.dnn.blobFromImage(face,1.0,(227, 227), self.MODEL_MEAN_VALUES,swapRB=False)
                # Get Gender
                self.sex=self.get_Gender(blob)
                # Get Age
                self.age=self.get_Age(blob)
                # Get Emotion
                self.emotion_text=self.get_Emotion(x1,x2,y1,y2,gray)
                # Change Color
                self.bounder_color=self.change_Color(self.emotion_text)
                # Draw Labels
                self.draw_Labels(x1,x2,y1,y2,frame)
                # Show Result
#                cv2.imshow("2019SJTU SummerProject", frame)

            return frame
    # Draw Labels
    def draw_Labels(self,left,right,top,bottom,frame):
        cv2.rectangle(frame, (left, top), (right, bottom),self.bounder_color, 4)
        cv2.rectangle(frame,(left, bottom-30),(right, bottom+100),self.bounder_color,cv2.FILLED)
        cv2.putText(frame,self.name+':'+str(self.accuracy),(left+5,bottom-5),self.font,1.0, self.text_color, 2)
        cv2.putText(frame,'age:'+str(self.age),(left+5,bottom+25),self.font, 1.0,self.text_color, 2)
        cv2.putText(frame,'sex:'+str(self.sex), (left+5, bottom+55), self.font, 1.0,self.text_color, 2)
        if self.mode==2:
            cv2.putText(frame,self.emotion_text,(left+5,bottom+85),self.font,1.0,self.text_color,2)
        if self.mode==1:
            cv2.putText(frame,self.lock, (left+5, bottom+85), self.font, 1.0,self.text_color, 2)
        return
    # Change Color###
    def change_Color(self,emotion_text):
        if emotion_text == 'angry':
            color = self.emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = self.emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = self.emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = self.emotion_probability * np.asarray((0, 255, 255))
        else:
            color = self.emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()
        return color
    # Start
    def Start(self):
        self.Console("Starting...")
        self.init_Mask()
        while self.listener:
            frame=self.read_Data()
            frame=self.make_Mask(frame)
#            self.get_Detections(frame)
            self.show_Mode(frame)
            self.Listener_Keys()
#            cv2.imshow("2019SJTU SummerProject", frame)
    # Mask Effect
    def make_Mask(self,frame):
#        self.Console("Making Mask Effect...")
        frame = resize(frame, width=self.max_width)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(img_gray, 0)
        faces = self.get_Orientation(rects, img_gray)
        draw_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.doing:
            self.Draw(draw_img, faces)
            self.animation_time += self.speed
            self.save_Data(draw_img)
            if self.animation_time > self.duration:
                self.doing = False
                self.animation_time = 0
            else:
                frame = cv2.cvtColor(np.asarray(draw_img), cv2.COLOR_RGB2BGR)
        return frame
    # Keys Listener
    def Listener_Keys(self):
        key=cv2.waitKey(1) & 0xFF
        if key==ord("q"):
            self.listener = False
            self.Console("Exiting...")
            sleep(1)
            self.Exit()
        if key==ord("z"):
            self.doing = not self.doing
            self.Console("Making Mask Effect...")

        if key==ord("c"):
            self.Capture()
        if key==ord("x"):
            self.change_Mode()
    # Capture
    def Capture(self):
        frame=self.read_Data()
        cv2.imwrite("capture/Capture" + '_' + str(self.count) + ".jpg", frame)
        self.Console("Capturing...")
        self.count+=1
        return
    # Change Mode
    def change_Mode(self):
        if self.mode==1:
            self.mode+=1
            self.Console("Entering Mode02:Face Detection and Capture...")
            return
        elif self.mode==2:
            self.mode+=1
            self.Console("Entering Mode03:Face Masking...")
            return
        elif self.mode==3:
            self.mode=1
            self.Console("Entering Mode01:Face Recognition...")
            return
    
    # Show Mode
    def show_Mode(self,frame):
        mode_console='Mode01:Face Recognition'
        if self.mode==1:
#            self.Console("Entering Mode01:Face Recognition...")
            msode_console='Mode01:Face Recognition'
            self.detect_Faces(frame)
            self.show_Lock(frame)
        elif self.mode==2:
#            self.Console("Entering Mode02:Face Detection and Capture...")
            mode_console='Mode02:Face Detection and Capture'
            self.get_Detections(frame)
        elif self.mode==3:
#            self.Console("Entering Mode03:Face Masking...")
            mode_console='Mode03:Face Masking'

        cv2.putText(frame,mode_console,(10,20),self.font,1.0, self.text_color, 2)
        cv2.imshow("2019SJTU SummerProject", frame)

        return
    # Show Lock
    def show_Lock(self,frame):
        cv2.putText(frame,self.lock,(10,300),self.font,1.0, self.text_color, 2)
        return
    # Draw
    def Draw(self, draw_img, faces):
        for face in faces:
            if self.animation_time < self.duration - self.fixed_time:
                current_x = int(face["glasses"]["pos"][0])
                current_y = int(face["glasses"]["pos"][1] * self.animation_time / (self.duration - self.fixed_time))
                draw_img.paste(face["glasses"]["image"], (current_x, current_y), face["glasses"]["image"])

                cigarette_x = int(face["cigarette"]["pos"][0])
                cigarette_y = int(face["cigarette"]["pos"][1] * self.animation_time / (self.duration - self.fixed_time))
                draw_img.paste(face["cigarette"]["image"], (cigarette_x, cigarette_y),
                               face["cigarette"]["image"])
            else:
                draw_img.paste(face["glasses"]["image"], face["glasses"]["pos"], face["glasses"]["image"])
                draw_img.paste(face["cigarette"]["image"], face["cigarette"]["pos"], face["cigarette"]["image"])
                draw_img.paste(self.text, (75, draw_img.height // 2 + 128), self.text)
    # Console
    def Console(cls, s):
        print("{}".format(s))
    # Exit
    def Exit(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    MP = MainProgram()
    MP.Start()
