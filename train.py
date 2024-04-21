import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import json

def create_custom_model(num_classes):
    return Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='same', bias_regularizer=l2(0.01), kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same', bias_regularizer=l2(0.01), kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        Conv2D(128, (3, 3), activation='relu', padding='same', bias_regularizer=l2(0.01), kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(256, activation='relu', bias_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

class MetricsCallback(Callback):
    def __init__(self, validation_generator):
        super().__init__()
        self.validation_generator = validation_generator
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
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
    parser.add_argument('-model', type=str, required=True, help='Model to train: custom')
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
        color_mode='grayscale',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(48, 48),
        batch_size=args.bs,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation'
    )

    num_classes = train_generator.num_classes
    model = create_custom_model(num_classes)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001),
        MetricsCallback(validation_generator),
        CSVLogger(os.path.join('models', args.fn + '.log'))
    ]

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, args.fn + '.h5'))

    hyperparameters = {
        'model_name': args.fn,
        'epochs': args.epochs,
        'batch_size': args.bs,
        'learning_rate': args.lr,
        'val_f1_scores': callbacks[1].val_f1s,
        'val_recall_scores': callbacks[1].val_recalls,
        'val_precision_scores': callbacks[1].val_precisions
    }
    with open(os.path.join(model_dir, 'training_parameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(model_dir, 'training_validation_curves.png'))
    plt.close()

if __name__ == '__main__':
    main()
