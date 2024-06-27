import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16(weights='imagenet')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (224, 224))
        face_resized_color = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        face_resized_color = cv2.resize(face_resized_color, (224, 224))
        face_resized_color = np.expand_dims(face_resized_color, axis=0)
        face_resized_color = face_resized_color / 255.0
        predictions = model.predict(face_resized_color)[0]
        precision = np.max(predictions)
        emotion_label = emotions[np.argmax(predictions)]
        cv2.putText(frame, emotion_label, (x - 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    

        # text = ''
        # y_offset = y
        # for emotion, prediction in zip(emotions, predictions):
        #     text = f'{emotion}: {prediction.item():.2f}'
        #     cv2.putText(frame, text, (x - 120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #     y_offset += 20
    
    cv2.imshow('Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()