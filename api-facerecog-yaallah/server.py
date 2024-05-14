from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import zipfile
import os
import numpy as np
import random
import string
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras.models import load_model
from numpy import asarray
import pickle
import shutil
import uuid

from functools import partial

from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import add
import tensorflow.keras.backend as K


app = FastAPI()

def scaling(x, scale):
    return x * scale


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(axis=bn_axis, momentum=0.995, epsilon=0.001,
                               scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x


def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == 'Block35':
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 128, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 128, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 192, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 192, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Block35", "Block17" or "Block8", '
                         'but got: ' + str(block_type))

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    up = Lambda(scaling,
                output_shape=K.int_shape(up)[1:],
                arguments={'scale': scale})(up)
    x = add([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x


def InceptionResNetV1(input_shape=(160, 160, 3),
                      classes=128,
                      dropout_keep_prob=0.8,
                      weights_path=None):
    inputs = Input(shape=input_shape)
    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
    x = conv2d_bn(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')

    # 5x Block35 (Inception-ResNet-A block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x,
                                    scale=0.17,
                                    block_type='Block35',
                                    block_idx=block_idx)

    # Mixed 6a (Reduction-A block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 2))(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    # 10x Block17 (Inception-ResNet-B block):
    for block_idx in range(1, 11):
        x = _inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='Block17',
                                    block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 2))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 3))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    # 5x Block8 (Inception-ResNet-C block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='Block8',
                                    block_idx=block_idx)
    x = _inception_resnet_block(x,
                                scale=1.,
                                activation=None,
                                block_type='Block8',
                                block_idx=6)

    # Classification block
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
    # Bottleneck
    x = Dense(classes, use_bias=False, name='Bottleneck')(x)
    bn_name = _generate_layer_name('BatchNorm', prefix='Bottleneck')
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
                           name=bn_name)(x)

    # Create model
    model = Model(inputs, x, name='inception_resnet_v1')
    if weights_path is not None:
        model.load_weights(weights_path)

    return model


mtcnn_detector = MTCNN()

weight = "facenet_keras_weights.h5"
model = InceptionResNetV1(weights_path = weight)

def detect_face(filename, required_size=(160, 160), normalize=True):

    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)


    detector = MTCNN()


    results = detector.detect_faces(pixels)
    if results:

        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = pixels[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)

        if normalize:
            face_array = (face_array - face_array.mean()) / face_array.std()

        return face_array

    return None


# known_faces_encodings = []
# known_faces_ids = []
# known_faces_path = "database/"
# valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

# for filename in os.listdir(known_faces_path):
#     if filename.endswith(valid_extensions):
#         full_path = os.path.join(known_faces_path, filename)
#         face = detect_face(full_path, normalize=True)
#         if face is not None:
#             feature_vector = model.predict(face.reshape(1, 160, 160, 3))
#             feature_vector /= np.sqrt(np.sum(feature_vector**2))
#             known_faces_encodings.append(feature_vector)
#             label = filename.split('.')[0]
#             known_faces_ids.append(label)

# known_faces_encodings = np.array(known_faces_encodings).reshape(len(known_faces_encodings), 128)
# known_faces_ids = np.array(known_faces_ids)

# # Function to recognize a face
# def recognize(img, known_faces_encodings, known_faces_ids, threshold=0.75):
#     enc = model.predict(img.reshape(1, 160, 160, 3))
#     enc /= np.sqrt(np.sum(enc**2))
#     scores = np.sqrt(np.sum((enc - known_faces_encodings)**2, axis=1))
#     match = np.argmin(scores)
#     if scores[match] > threshold:
#         return ("UNKNOWN", scores[match])
#     else:
#         return (known_faces_ids[match], scores[match])

# def test_image_recognition(image_path, known_faces_encodings, known_faces_ids, threshold=1):
#     face = detect_face(image_path)
#     if face is not None:
#         label, score = recognize(face, known_faces_encodings, known_faces_ids, threshold)
#         print(f"Detected face as {label} with a score of {score}")
#     else:
#         print("No face detected in the image.")
        
# test_image_recognition('cropped/Photo.png', known_faces_encodings, known_faces_ids)

def save_encodings(encodings, labels, class_name):
    encodings_dir = "encodings"
    os.makedirs(encodings_dir, exist_ok=True)
    
    file_path = os.path.join(encodings_dir, f'encodings_{class_name}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump({'labels': labels, 'encodings': encodings}, f)

    
def recognize(img, known_faces_encodings, known_faces_ids, threshold=0.75):
    img = img.reshape(1, 160, 160, 3)
    enc = model.predict(img)
    enc /= np.sqrt(np.sum(enc**2))
    scores = np.sqrt(np.sum((enc - known_faces_encodings)**2, axis=1))
    match = np.argmin(scores)
    if scores[match] > threshold:
        return ("UNKNOWN", float(scores[match])) 
    else:
        return (known_faces_ids[match], float(scores[match]))  


    
valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

@app.post("/upload-class-photos/")
async def upload_class_photos(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .zip file.")

    temp_dir = "temp_photos"
    class_name = str(uuid.uuid4())
    class_dir = os.path.join(temp_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    try:
        with open(f"{class_dir}.zip", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(f"{class_dir}.zip", 'r') as zip_ref:
            zip_ref.extractall(class_dir)

        print("Files in directory:", os.listdir(class_dir))  

        foto_dir = os.path.join(class_dir, "foto")
        if os.path.exists(foto_dir):
            files_to_process = os.listdir(foto_dir)
        else:
            files_to_process = os.listdir(class_dir)

        print("Files to process:", files_to_process)  

        known_faces_encodings = []
        known_faces_ids = []
        file_count = 0

        for filename in files_to_process:
            if filename == '__MACOSX':
                continue  
            full_path = os.path.join(foto_dir, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(valid_extensions):
                file_count += 1
                face = detect_face(full_path, normalize=True)
                if face is not None:
                    print(f"Detected face shape: {face.shape}")
                    face_array = face.reshape(1, 160, 160, 3)
                    feature_vector = model.predict(face_array)
                    feature_vector /= np.sqrt(np.sum(feature_vector**2))
                    known_faces_encodings.append(feature_vector.flatten())
                    label = filename.split('.')[0]
                    known_faces_ids.append(label)
                else:
                    print(f"No face detected in {filename}")

        print(f"Total files processed: {file_count}")
        print(f"Total faces encoded: {len(known_faces_ids)}")

        save_encodings(known_faces_encodings, known_faces_ids, class_name)

    finally:
        shutil.rmtree(class_dir)
        os.remove(f"{class_dir}.zip")

    return {"message": "Photos processed successfully", "class_id": class_name}


@app.post("/recognize-face/")
async def recognize_face(image: UploadFile = File(...), class_id: str = File(...)):
    encodings_path = f'encodings/encodings_{class_id}.pkl'
    if not os.path.exists(encodings_path):
        raise HTTPException(status_code=404, detail="Class ID not found.")

    try:
        with open(encodings_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load encodings: {str(e)}")

    temp_file_path = f'temp_{image.filename}'
    with open(temp_file_path, 'wb') as buffer:
        shutil.copyfileobj(image.file, buffer)

    face = detect_face(temp_file_path, normalize=True)
    os.remove(temp_file_path)  

    if face is not None:
        label, score = recognize(face, np.array(data['encodings']), data['labels'])
        return {"label": label, "score": score}
    else:
        return JSONResponse(status_code=400, content={"message": "No face detected in the image."})

    
    
# import pickle

# def load_and_display_pkl_contents(file_path):
#     try:
#         with open(file_path, 'rb') as file:
#             data = pickle.load(file)
#             print("Contents of the .pkl file:")
#             print("Labels:")
#             for label in data['labels']:
#                 print(label)
#             print("\nEncodings:")
#             for encoding in data['encodings']:
#                 print(encoding)
#             return data
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#     except Exception as e:
#         print(f"An error occurred while loading the .pkl file: {e}")

# file_path = 'encodings_13a1ab66-86c9-4903-9a07-32edc5056d43.pkl'  
# load_and_display_pkl_contents(file_path)

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)