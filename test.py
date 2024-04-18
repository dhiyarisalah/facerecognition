import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

# Load your trained Face Recognition model
model_path = './models/35_custom_ori.h5'
face_recognition_model = tf.keras.models.load_model(model_path)

# Load labels from a CSV file
def load_labels(label_file):
    labels_df = pd.read_csv(label_file)
    labels = labels_df['label'].tolist()  # Ensure your CSV has a column named 'label'
    return labels

labels = load_labels('./labels.csv')  # Path to your labels file

# Haar Cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process each face
def process_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))  # Resize to match the input size of your model
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0) / 255.0
    prediction = face_recognition_model.predict(face_img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_index, labels[class_index] if confidence > 70 else "Unknown", confidence

# Function to process an image
def process_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face found
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        class_index, label, confidence = process_face(face_roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print the label, index, and confidence to the console
        print(f"Detected: {label} (Index: {class_index}, Confidence: {confidence:.2f}%)")

    # Show the image
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage: Provide the path to your image file
image_path = './test/glover.jpg'
process_image(image_path)



# # Load your trained Face Recognition model
# model_path = './models/35_custom_ori.h5'
# face_recognition_model = tf.keras.models.load_model(model_path)

# # Load labels from a CSV file
# def load_labels(label_file):
#     labels_df = pd.read_csv(label_file)
#     labels = labels_df['label'].tolist()  # Ensure your CSV has a column named 'label'
#     return labels

# labels = load_labels('./labels.csv')  # Path to your labels file

# # Haar Cascade for detecting faces
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Create a VideoCapture object to access the webcam
# cap = cv2.VideoCapture(1)  # Adjusted to default camera, change it if using another camera

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Function to process each face
# def process_face(face_img):
#     face_img = cv2.resize(face_img, (48, 48))  # Resize to match the input size of your model
#     face_img = np.expand_dims(face_img, axis=-1)
#     face_img = np.expand_dims(face_img, axis=0) / 255.0
#     prediction = face_recognition_model.predict(face_img)
#     class_index = np.argmax(prediction)
#     confidence = np.max(prediction) * 100
#     return labels[class_index] if confidence > 70 else "Unknown", confidence, class_index

# # Variables for FPS calculation
# frame_count = 0
# start_time = time.time()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     annotations = []

#     for (x, y, w, h) in faces:
#         # Extract face ROI
#         face_roi = gray[y:y+h, x:x+w]
#         label, confidence, class_index = process_face(face_roi)
#         annotations.append((x, y, w, h, label, confidence))

#     # Draw annotations
#     for (x, y, w, h, label, confidence) in annotations:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f"{label} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Calculate FPS
#     frame_count += 1
#     elapsed_time = time.time() - start_time
#     fps = frame_count / elapsed_time
#     cpu_usage = psutil.cpu_percent()

#     # Display FPS and CPU usage on the frame
#     cv2.putText(frame, f"FPS: {fps:.2f}, CPU: {cpu_usage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Face Recognition', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and destroy all windows
# cap.release()
# cv2.destroyAllWindows()
