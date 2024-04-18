import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class Preprocessing:
    def __init__(self, base_save_dir="../preprocessed"):
        self.detector = MTCNN()
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],  # Adjust brightness
            channel_shift_range=20.0  # Randomly shift color channels
        )
        self.base_save_dir = base_save_dir
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)

    def detect_and_align_face(self, img, size=(224, 224)):
        detections = self.detector.detect_faces(img)
        if detections:
            x, y, width, height = detections[0]['box']
            keypoints = detections[0]['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            eye_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx)) - 180

            # Rotate image to align eyes horizontally
            M = cv2.getRotationMatrix2D(eye_center, angle, 1)
            aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            # Crop aligned face
            x, y, w, h = cv2.boundingRect(np.array([[(x, y), (x+width, y+height)]]))
            cropped_img = aligned_img[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, size)
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            return gray_img
        return None  # Return None if no face detected

    def load_dataset(self, dataset_folder, size=(224, 224)):
        names, images = [], []
        for dirpath, dirnames, filenames in os.walk(dataset_folder):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(dirpath, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = self.detect_and_align_face(img, size=size)
                        if img is not None:
                            images.append(img)
                            label = os.path.basename(dirpath)
                            names.append(label)
        return names, images

    def image_augmentator(self, images, names):
        augmented_images = []
        augmented_names = []
        for i, (img, label) in enumerate(zip(images, names)):
            save_path = os.path.join(self.base_save_dir, label)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            num_existing = len([name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name))])
            num_augmentations = max(1, 50 - num_existing)  # Adjust as needed
            img = img.reshape((1,) + img.shape + (1,))  # Add batch and channel dimensions
            for j, batch in enumerate(self.datagen.flow(img, batch_size=1)):
                aug_img = batch[0].astype(np.uint8).reshape(img.shape[1:3])
                cv2.imwrite(os.path.join(save_path, f"{label}_{i}_{j+1}.png"), aug_img)
                if j >= num_augmentations - 1:
                    break
            augmented_images.append(img.reshape(img.shape[1:3]))  # Append original image
            augmented_names.append(label)
        return augmented_names, augmented_images

    def convert_categorical(self, names):
        le = LabelEncoder()
        name_vec = le.fit_transform(names)
        categorical_name_vec = to_categorical(name_vec)
        return le.classes_, categorical_name_vec

    def split_dataset(self, images, labels, test_size=0.20):
        return train_test_split(np.array(images, dtype=np.float32), labels, test_size=test_size, random_state=42)

# Usage
preprocessing = Preprocessing()
names, images = preprocessing.load_dataset("../ori")
names, images = preprocessing.image_augmentator(images, names)
