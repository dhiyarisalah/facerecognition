import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import argparse
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score

def create_custom_model(num_classes):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# Adapt to use num_classes for pretrained models
def create_vgg16_model(num_classes):
    base_model = tf.keras.applications.VGG16(weights='imagenet', input_shape=(48, 48, 3), include_top=False)
    base_model.trainable = False
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

def create_resnet_model(num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(48, 48, 3), include_top=False)
    base_model.trainable = False
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# Custom callback for additional metrics
class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
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
    base_dir = '../'  
    dataset_dir = os.path.join(base_dir, 'preprocessed_' + args.dataset)

    train_dir = dataset_dir
    test_dir = dataset_dir  

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Using 20% of data for validation
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=args.bs,
        class_mode='categorical',
        color_mode='rgb' if args.model in ['vgg16', 'resnet'] else 'grayscale',
        subset='training'  
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
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

    callbacks = [
        ModelCheckpoint(os.path.join('../models', args.fn + '.h5'), save_best_only=True),
        ReduceLROnPlateau(),
        MetricsCallback(validation_data=(validation_generator.next()[0], validation_generator.next()[1])),
        CSVLogger(os.path.join('../logs', args.fn + '.log'))
    ]

    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

if __name__ == '__main__':
    main()
