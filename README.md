# 😊 Real-Time Emotion Detection via Facial Recognition

A deep learning-based application that detects and classifies human emotions in real time using facial expressions captured from a webcam. This project explores the intersection of artificial intelligence and human emotion analysis.

---

## 📌 Overview

Emotions play a central role in human interaction. This project aims to replicate the human ability to recognize facial expressions in machines using deep learning techniques, enabling real-time analysis of emotions such as:

- 😄 Happy
- 😢 Sad
- 😠 Angry
- 😱 Fear
- 😮 Surprise
- 🤢 Disgust
- 😐 Neutral

---

## 🎯 Objectives

- Capture facial expressions in real time from a webcam.
- Analyze the face and classify the emotion among 7 predefined categories.
- Display the detected emotion visually on the video stream.

---

## 🧠 Methodology

### 1. 🗃️ Dataset & Preprocessing

- **Dataset:** [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Preprocessing:**
  - Resize images to 48x48
  - Convert to grayscale
  - Normalize pixel values

### 2. 🧱 Model Architecture

- **Transfer Learning:** MobileNetV2
- **Custom Classification Head:**
    ```python
    final_output = layers.Dense(128)(base_output)
    final_output = layers.Activation('relu')(final_output)
    final_output = layers.Dense(64)(final_output)
    final_output = layers.Activation('relu')(final_output)
    final_output = layers.Dense(7, activation='softmax')(final_output)
    ```
- Trained using Google Colab with GPU acceleration.

### 3. ⚙️ Deployment

- Real-time image capture via OpenCV
- Face detection using Haar Cascade
- Emotion prediction with TensorFlow/Keras
- Interface built in Jupyter Notebook

---

## ✅ Results

- **Accuracy:** 97.7% on validation data.
- **Speed:** Average processing time per frame ~101 ms.
- **Performance:** Robust under standard lighting and camera conditions.

---

## 🧩 Challenges

- Subtle differences between emotions (e.g., neutral vs. sad)
- Image quality variations due to lighting or blurriness
- Difficulty handling partially occluded or off-angle faces

---

## 🖥️ Interface Demo

- Live webcam preview
- Emotion detected and labeled on the video stream
- Top 3 predictions shown with confidence scores

---

## 🌍 Potential Applications

- 🎯 **Marketing:** Real-time customer feedback through emotion analysis
- 🧠 **Mental health:** Assist therapists in emotion monitoring
- 🤖 **Virtual assistants:** Improve responses by detecting user emotions
- 🏫 **E-learning platforms:** Adapt teaching pace to learner mood
- 🕵️‍♀️ **Security:** Suspicious behavior detection in surveillance systems

---

## 🚀 Future Work

- Integrate voice and text sentiment analysis for multimodal emotion detection
- Develop a mobile app for real-world use
- Improve model robustness in diverse environments

---

## 👥 Authors

- **Hiba Almizouni**

Supervised by **Mr. Walid Chainbi**  
📍 École Nationale d’Ingénieurs de Sousse, Département Informatique Appliquée

---

## 📚 Bibliography

- FER-2013 Dataset – Kaggle
- *Machine Learning For Absolute Beginners* by Oliver Theobald
- *Hands-on ML with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron
- NVIDIA Deep Learning Institute
- GitHub & ChatGPT (OpenAI)
