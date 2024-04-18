import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import json

def create_custom_model(num_classes):
    return Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

def create_vgg16_model(num_classes):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    base_model.trainable = False
    return Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def create_resnet_model(num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    base_model.trainable = False
    return Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

class MetricsCallback(Callback):
    def __init__(self, validation_generator):
        super().__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        self.validation_generator.reset()
        val_predict = []
        val_targ = []

        for x_val, y_val in self.validation_generator:
            val_predict.extend(np.argmax(self.model.predict(x_val), axis=-1))
            val_targ.extend(np.argmax(y_val, axis=-1))
            if len(val_predict) >= self.validation_generator.samples:
                break

        _val_precision = precision_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_f1 = f1_score(val_targ, val_predict, average='macro')

        logs['val_precision'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        print(f' — val_precision: {_val_precision} — val_recall: {_val_recall} — val_f1: {_val_f1}')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('-model', type=str, required=True, help='Model to train: custom, vgg16, resnet')
    parser.add_argument('-dataset', type=str, required=True, help='Dataset to use: ori, masked')
    parser.add_argument('-epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-bs', type=int, required=True, help='Batch size')
    parser.add_argument('-lr', type=float, required=True, help='Learning rate')
    parser.add_argument('-fn', type=str, required=True, help='Model file name')
    return parser.parse_args()

def main():
    args = parse_arguments()
    dataset_dir = os.path.join('../', 'preprocessed_' + args.dataset)

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(48, 48),
        batch_size=args.bs,
        class_mode='categorical',
        color_mode='rgb' if args.model in ['vgg16', 'resnet'] else 'grayscale',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(48, 48),
        batch_size=args.bs,
        class_mode='categorical',
        color_mode='rgb' if args.model in ['vgg16', 'resnet'] else 'grayscale',
        subset='validation'
    )

    num_classes = train_generator.num_classes
    if args.model == 'custom':
        model = create_custom_model(num_classes)
    elif args.model == 'vgg16':
        model = create_vgg16_model(num_classes)
    elif args.model == 'resnet':
        model = create_resnet_model(num_classes)
    else:
        raise ValueError("Invalid model type. Choose 'custom', 'vgg16', or 'resnet'.")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Prepare directories for saving models and logs
    model_dir = os.path.join('../models')
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        ReduceLROnPlateau(),
        MetricsCallback(validation_generator),
        CSVLogger(os.path.join(model_dir, args.fn + '.log'))
    ]

    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Manually save the model to ensure it's in .h5 format
    model.save(os.path.join(model_dir, args.fn + '.h5'))

if __name__ == '__main__':
    main()
