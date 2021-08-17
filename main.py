from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import os


class RecognizerGUI:
    def __init__(self, root):
        root.title("Face Recogniser and Trainer v1.0")
        root.minsize(1000, 600)
        root.maxsize(1000, 600)
        root.resizable(False, False)
        self.face_id = 0
        self.count = 0

        user_ui_left = Frame(root, width=200, height=600, bg="grey")
        user_ui_left.anchor = LEFT
        user_ui_left.grid(row=0, column=0, padx=10, pady=5)

        user_ui_right = Frame(root, width=200, height=600, bg="grey")
        user_ui_right.anchor = RIGHT
        user_ui_right.grid(row=0, column=2, padx=10, pady=5)

        self.camera_box = Label(root, bg="grey")
        self.camera_box.anchor = CENTER
        self.camera_box.grid(row=0, column=1, padx=0, pady=0)
        msg_box_content = "Steps\n 1. Make sure all directories exist\n 2. Click AddNewFace  \n " \
                          "3. Train the model on new datasets\n 4. Test the model by clicking TEST "
        msg_box = Label(root, width=28, height=30, bg="white", text=msg_box_content)
        msg_box.anchor = RIGHT
        msg_box.grid(row=0, column=2, padx=2, pady=2)

        AddNewFace = Button(user_ui_left, text="Add New Face", width=20, height=1, command=self.video_stream, bg="red")
        Train = Button(user_ui_left, text="Train Model", width=20, height=1, command=self.train, bg='white')
        Test = Button(user_ui_left, text="Test", width=20, height=1, command=self.get_inference, bg='grey')
        Check = Button(user_ui_left, text="Check", width=20, height=1, command=self.do_nothing, bg='green')

        AddNewFace.grid(row=0, column=0, sticky=W, pady=2, padx=2)
        Train.grid(row=1, column=0, sticky=W, pady=2, padx=2)
        Test.grid(row=2, column=0, sticky=W, pady=2, padx=2)
        Check.grid(row=3, column=0, sticky=W, pady=2, padx=2)

        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def do_nothing(self):
        pass  # this is not supposed to do any thing

    def video_stream(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 80)  # set video width
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)  # set video height
        self.count = 0
        self.face_id = int(input("Enter face ID:"))

        def stream_it():
            _, frame = cam.read()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.analyze_and_save_image(cv2image)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_box.imgtk = imgtk
            self.camera_box.configure(image=imgtk)
            self.count += 1
            if self.count >= 100:
                print("\n Closing data set generator and cleaning up things!")
                cam.release()
                cv2.destroyAllWindows()
                return
            self.camera_box.after(1, stream_it)

        stream_it()

    def analyze_and_save_image(self, img):
        gray = img
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imwrite("dataset/User." + str(self.face_id) + '.' + str(self.count) + ".jpg", gray[y:y + h, x:x + w])

    def train(self):
        path = 'dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
            return faceSamples, ids

        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    def get_inference(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 80)  # set video width
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)  # set video height
        self.count = 0

        def stream_it_inference():
            _, frame = cam.read()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.make_inference(cv2image, recognizer, names)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_box.imgtk = imgtk
            self.camera_box.configure(image=imgtk)
            self.count += 1
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                print("Recognition service stopped!")
                cam.release()
                return
            self.camera_box.after(1, stream_it_inference)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        names = ['None', 'Shaheen', 'Person1', 'Person2', 'Person3', 'Person4']
        stream_it_inference()

    def make_inference(self, img, recognizer, names):
        minW = 64.0
        minH = 48.0
        gray = img
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)


def close():
    window.withdraw()


window = Tk()
window.bind('<Escape>', close)
app = RecognizerGUI(window)
window.mainloop()
