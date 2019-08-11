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
        
        self.saved = saved  # Save Image
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
        self.max_width = 500  # Size
        self.deal, self.text, self.cigarette = None, None, None  # Masks
    
        # Load Picture
        self.Jessica_image = face_recognition.load_image_file("Jessica.jpeg")
        self.Jessica_face_encoding = face_recognition.face_encodings(self.Jessica_image)[0]
        self.George_image = face_recognition.load_image_file("George.jpg")
        self.George_face_encoding = face_recognition.face_encodings(self.George_image)[0]
        # Labels
        self.known_face_encodings = [self.Jessica_face_encoding,self.George_face_encoding]
        self.known_face_names = ["Jessica.C","George.T"]
        self.ages =[20,21]
        self.sexs=['F','M']

        # Variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.text_color=(255,255,255)
        self.bounder_color=(0,0,255)
    
    # Draw Labels
    def draw_Labels(self,left,right,top,bottom,frame):
#        cv2.rectangle(frame, (left, top), (right, bottom),self.bounder_color, 2)
#        cv2.rectangle(frame,(left, bottom-35),(right, bottom+100),self.bounder_color,cv2.FILLED)
        cv2.putText(frame,self.name+':'+str(self.accuracy),(left + 6, bottom - 5),self.font,1.0, self.text_color, 2)
#        cv2.putText(frame,'age:'+str(self.age),(left+6,bottom+25),self.font, 1.0,self.text_color, 2)
#        cv2.putText(frame,str(self.sex), (left + 6, bottom+55), self.font, 1.0,self.text_color, 2)
#        cv2.putText(frame,self.lock, (left + 6, bottom+85), self.font, 1.0,self.text_color, 2)
        return
    
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
    # Start
    def Start(self):
        self.Console("Starting...")
        self.init_Mask()
        while self.listener:
            frame = self.read_Data()
            frame = resize(frame, height=800,width=600)
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
            self.detect_Faces(frame)
            cv2.imshow("Mode 03", frame)
            self.Listener_Keys()
    # Keys Listener
    def Listener_Keys(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.listener = False
            self.Console("Exiting...")
            sleep(1)
            self.Exit()
        if key == ord("d"):
            self.doing = not self.doing
    # Exit
    def Exit(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
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
    def Console(cls, s):
        print("{} !".format(s))


if __name__ == '__main__':
    MP = MainProgram()
    MP.Start()
