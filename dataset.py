import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class Preprocessing:
    def __init__(self, base_save_dir="../preprocessed"):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.datagen =  ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2], 
            channel_shift_range=10.0
        )

        self.base_save_dir = base_save_dir
        os.makedirs(self.base_save_dir, exist_ok=True)

    def detect_and_align_face(self, img, size=(160, 160)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Histogram equalization
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if faces is not None and len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])  # Choose the largest face
            cropped_img = img[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, size)
            return resized_img
        return None

    def load_dataset(self, dataset_folder, size=(160, 160)):
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
        augmented_images, augmented_names = [], []
        for img, label in zip(images, names):
            save_path = os.path.join(self.base_save_dir, label)
            os.makedirs(save_path, exist_ok=True)
            img = img.reshape((1,) + img.shape)  
            for batch in self.datagen.flow(img, batch_size=1, save_to_dir=save_path, save_prefix=f"{label}_", save_format='png'):
                augmented_images.append(batch[0].astype(np.uint8))
                augmented_names.append(label)
                if len(augmented_images) % 50 == 0:
                    break
        return augmented_names, augmented_images

    def convert_categorical(self, names):
        le = LabelEncoder()
        name_vec = le.fit_transform(names)
        categorical_name_vec = to_categorical(name_vec)
        return le.classes_, categorical_name_vec

    def split_dataset(self, images, labels, test_size=0.20):
        return train_test_split(np.array(images, dtype=np.float32) / 255.0, labels, test_size=test_size, random_state=42)

# Usage
preprocessing = Preprocessing()
names, images = preprocessing.load_dataset("../ori")
names, images = preprocessing.image_augmentator(images, names)
