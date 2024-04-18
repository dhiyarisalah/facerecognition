import os
import cv2
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class Preprocessing():
    def __init__(self, base_save_dir="../preprocessed"):
        self.detector = MTCNN()
        self.datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],  # Adjust brightness, 0.8 to 1.2 range
        channel_shift_range=20.0  # Randomly shift color channels
        )
        self.base_save_dir = base_save_dir
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)

    def detect_face(self, img, size=(224, 224)):
        result = self.detector.detect_faces(img)
        if len(result) == 0:
            return None
        x, y, width, height = result[0]['box']
        face = img[y:y+height, x:x+width]
        face = cv2.resize(face, size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return face

    def load_dataset(self, dataset_folder, size=(224, 224)):
        names, images = [], []
        for dirpath, dirnames, filenames in os.walk(dataset_folder):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(dirpath, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = self.detect_face(img, size=size)
                        if img is not None:
                            images.append(img)
                            label = os.path.basename(dirpath)
                            names.append(label)
        return names, images

    def img_augmentation(self, img, label, index):
        save_path = os.path.join(self.base_save_dir, label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Determine the number of augmentations needed based on existing samples
        num_existing = len([name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name))])
        num_augmentations = max(1, 50 - num_existing)  # Reduce number as the number of samples increases

        img = img.reshape((1,) + img.shape + (1,))
        for i, batch in enumerate(self.datagen.flow(img, batch_size=1)):
            aug_img = batch[0].astype(np.uint8).reshape(img.shape[1:3])
            cv2.imwrite(os.path.join(save_path, f"{label}_{index}_{i+1}.png"), aug_img)
            if i >= num_augmentations - 1:
                break

    def image_augmentator(self, images, names):
        augmented_images = []
        augmented_names = []
        for i, (img, label) in enumerate(zip(images, names)):
            self.img_augmentation(img, label, i)
            augmented_images.append(img)  # Append original image for reference
            augmented_names.append(label)
        return augmented_names, augmented_images

    def convert_categorical(self, names):
        le = LabelEncoder()
        name_vec = le.fit_transform(names)
        categorical_name_vec = to_categorical(name_vec)
        return le.classes_, categorical_name_vec

    def split_dataset(self, images, labels, test_size=0.20):
        return train_test_split(np.array(images, dtype=np.float32), labels, test_size=test_size, random_state=42)
    
preprocessing = Preprocessing()
names, images = preprocessing.load_dataset("ori")
names, images = preprocessing.image_augmentator(images, names)