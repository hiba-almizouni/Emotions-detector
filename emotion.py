import tkinter as tk
from tkinter import Label
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("my_modelfinale2.h5")  # Adjust path as needed
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# GUI Class
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.video_label = Label(root)
        self.video_label.pack()

        self.result_label = Label(root, text="", font=("Arial", 16))
        self.result_label.pack()

        self.start_btn = tk.Button(root, text="Start", command=self.start)
        self.start_btn.pack(pady=10)

        self.stop_btn = tk.Button(root, text="Stop", command=self.stop)
        self.stop_btn.pack(pady=10)

        self.cap = None
        self.running = False

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.update_frame).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
            self.video_label.config(image="")
            self.result_label.config(text="")

    def update_frame(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Display video
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Predict emotion
            face_img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (48, 48))
            face_array = np.expand_dims(face_img, axis=(0, -1)) / 255.0
            prediction = model.predict(face_array, verbose=0)[0]
            emotion = emotion_labels[np.argmax(prediction)]

            self.result_label.config(text=f"Emotion: {emotion}")
            self.root.update_idletasks()

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
