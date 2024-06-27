import cv2
import tensorflow as tf
from mtcnn import MTCNN


def draw_faces(image, faces):
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        for key, value in face['keypoints'].items():
            cv2.circle(image, value, 2, (0, 255, 0), 2)

model = tf.keras.models.load_model("./emotion_recognition_model.h5")
model.summary()

# Initialize MTCNN face detector
detector = MTCNN()

# Start video capture from the webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)
    
    # Draw bounding boxes and keypoints
    draw_faces(frame, faces)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()