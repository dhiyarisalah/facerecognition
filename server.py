import os
import io
import datetime
import zipfile
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from dataset import Preprocessing
from fastapi import FastAPI, UploadFile, File
import asyncio
import shutil

class TransferLearning:
    def __init__(self, model_type='custom', dim=128, use_augmentation=True, test_size=0.15, val_size=0.15, epoch=100, batch=32):
        self.model = None
        self.model_type = model_type
        self.dim = dim
        self.use_augmentation = use_augmentation
        self.test_size = test_size
        self.val_size = val_size
        self.epoch = epoch
        self.batch = batch
        self.is_running = False
        self.labels_dict = {}

    def init_model(self):
        if self.model_type == 'vgg16':
            self.model = self.adjust_vgg16_model()
        elif self.model_type == 'resnet':
            self.model = self.adjust_resnet_model()
        else:  # Default to custom model
            self.model = load_model('model-cnn.h5')
        print(f"Model initialized as {self.model_type}")

    async def run(self, zip_file: bytes):
        if not zip_file:
            return {"message": "No file uploaded"}

        try:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            zip_file = io.BytesIO(zip_file)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            self.is_running = True
            print(f"{self.get_time()} Start Transfer Learning")

            prepro = Preprocessing()
            names, images = prepro.load_dataset(dataset_folder=temp_dir)

            self.update_labels_file(names)

            print(f"{self.get_time()} Loaded dataset with {len(names)} samples and {len(self.labels_dict)} unique classes.")

            if self.use_augmentation:
                names, images = prepro.image_augmentator(images, names)
                print(f"{self.get_time()} Augmentation completed with {len(names)} samples.")

            labels = [self.labels_dict[name] for name in names]
            categorical_labels = to_categorical(labels, num_classes=len(self.labels_dict))

            x_train, x_test, y_train, y_test = prepro.split_dataset(images, categorical_labels, test_size=self.test_size)
            print(f"{self.get_time()} Dataset split into {len(x_train)} training and {len(x_test)} test samples.")

            self.model.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch, validation_split=self.val_size)
            new_model_name = "model_" + self.model_type.lower() + "_" + self.get_datetime_str() + ".h5"
            self.model.save(os.path.join(os.path.dirname(self.model_name), new_model_name))

            print(f"{self.get_time()} Training completed and model saved as {new_model_name}")

            shutil.rmtree(temp_dir)
            self.is_running = False
            return {"message": f"New model {new_model_name} created successfully"}

        except Exception as e:
            return {"message": f"Error: {str(e)}"}

    def adjust_vgg16_model(self):
        from tensorflow.keras.applications import VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.dim, self.dim, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(len(self.labels_dict), activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def adjust_resnet_model(self):
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.dim, self.dim, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(len(self.labels_dict), activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def update_labels_file(self, names):
        unique_names = set(names)
        new_labels = unique_names - set(self.labels_dict.keys())
        for label in new_labels:
            self.labels_dict[label] = len(self.labels_dict)
        print(f"{self.get_time()} Labels updated. Total classes: {len(self.labels_dict)}")

    def get_time(self):
        return f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}]"

    def get_datetime_str(self):
        return datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

app = FastAPI()

@app.post("/create_model")
async def create_model_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    transfer_learning = TransferLearning()
    transfer_learning.init_model()
    result = await asyncio.ensure_future(transfer_learning.run(content))
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
