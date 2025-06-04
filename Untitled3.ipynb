import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load model only once
new_model = load_model("C:/Users/pc/OneDrive/Bureau/emotion detector/my_modelfinale2.keras")

# Start webcam
cap = cv2.VideoCapture(0)

# Optional: Reduce resolution to improve speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ["Angry", "Disgust", "fear", "happy", "neutral", "sad", "Surprised"]

frame_count = 0
start_time = time.time()

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]
        final_image = cv2.resize(roi_color, (224, 224))
        final_image = np.expand_dims(final_image, axis=0) / 255.0

        predictions = new_model.predict(final_image, verbose=0).flatten()
        top_indices = np.argsort(predictions)[::-1][:3]

        main_emotion = emotion_labels[top_indices[0]]
        cv2.putText(frame, f'Main: {main_emotion}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for i, idx in enumerate(top_indices):
            text = f"{emotion_labels[idx]}: {predictions[idx]*100:.1f}%"
            cv2.putText(frame, text, (x, y + h + 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        process_frame(frame)

        # FPS counter
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Real-Time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
