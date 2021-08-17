# Simple-Face-Recognizer-Python-and-Yaml-
This is a Face Recognizer project.<br> 
GUI is used to make Adding New Faces, Training New Faces and Testing the Trained data simple.<br>
Implemented using Python Tkinter and OpenCv.<br>
Accuracy= ~70%.<br>


# Working
It captures the new face image from users webcam, finds the face in current frame, crop the image around the face and save it into "dataset" directory. Training is done using OpenCv face recognizer which saves the trained data in .yaml in "trainer" directory. Testing: Streams video from the webcam, detect faces and compares it with trained data.

root/<br>
  /main.py<br>
  /haarcascade_frontalface_default.xml<br>
  /dataset<br>
  /trainer<br>


# Note
Make sure to create required directories.<br>
Will make another version, but with a Tensorflow model and a fully working GUI!!!
