pip install deepface

import cv2
import os
from deepface import DeepFace

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

# Create a folder to save the captured faces if it doesn't exist
save_path = 'C:\\Users\\livia\\PycharmProjects\\facerecognition\\captured_faces'
os.makedirs(save_path, exist_ok=True)

# Global variable to keep track of the total number of faces detected
total_faces_detected = 0

def detect_bounding_box(vid):
    global total_faces_detected  # Use the global variable

    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    for i, (x, y, w, h) in enumerate(faces):
        face_img = vid[y:y + h, x:x + w]

        # Create a green rectangle with transparency
        alpha = 0  # You can adjust the transparency level
        overlay = vid.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, vid, 1 - alpha, 0, vid)

        # Generate a unique filename based on the total number of faces detected
        filename = os.path.join(save_path, f"face_{total_faces_detected}.png")

        cv2.imwrite(filename, face_img)  # Save the detected face

        # Increment the total number of faces detected
        total_faces_detected += 1

        # Perform face recognition using DeepFace
        try:
            result = DeepFace.verify(filename, detector_backend='mtcnn')
            print(f"Is the detected face recognized? {result['verified']}")
        except Exception as e:
            print(f"Error in face recognition: {e}")

    return faces

while True:
    result, video_frame = video_capture.read()
    if not result:
        break


    faces = detect_bounding_box(video_frame)

    cv2.imshow("My Face Detection Project", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

