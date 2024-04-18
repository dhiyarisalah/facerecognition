from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os

datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# The base directory where the train data is stored
base_train_dir = '../dataset/fer2013plus/train'

# The base directory where the augmented images will be saved
base_augmented_dir = '../dataset/augmented'

# List of emotions
# emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

emotions = ['surprise']

# Iterate over each emotion
for emotion in emotions:
    # Path to the specific emotion directory
    emotion_dir = os.path.join(base_train_dir, emotion)
    
    # Ensure the augmented directory for the current emotion exists
    augmented_emotion_dir = os.path.join(base_augmented_dir, emotion)
    if not os.path.exists(augmented_emotion_dir):
        os.makedirs(augmented_emotion_dir)
    
    # List all images in the emotion directory
    image_files = os.listdir(emotion_dir)
    
    # Iterate over each image file
    for image_file in image_files:
        # Full path to the image
        image_path = os.path.join(emotion_dir, image_file)
        
        # Read the image for augmentation
        try:
            x = io.imread(image_path)
            x = x.reshape((1, ) + x.shape)
        except:
            continue  # If an image can't be read, skip it

        # Initialize counter
        i = 0
        # Generate and save the augmented images
        for batch in datagen.flow(x, batch_size=16, save_to_dir=augmented_emotion_dir, save_prefix='aug', save_format='png'):
            i += 1
            if i > 3:  # Change this to how many augmentations you want per image
                break  # Stop after generating the desired number of augmented images


        print(f"Augmented images for {image_file} saved in {augmented_emotion_dir}")

# from keras.preprocessing.image import ImageDataGenerator
# from skimage import io
# import os

# datagen = ImageDataGenerator(
#     rotation_range=35,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # The base directory where the train data is stored
# base_train_dir = '../dataset/fer2013plus/train'

# # The base directory where the augmented images will be saved
# base_augmented_dir = '../dataset/augmented'

# # List of emotions
# emotions = ['anger']

# # Iterate over each emotion
# for emotion in emotions:
#     # Path to the specific emotion directory
#     emotion_dir = os.path.join(base_train_dir, emotion)
    
#     # Ensure the augmented directory for the current emotion exists
#     augmented_emotion_dir = os.path.join(base_augmented_dir, emotion)
#     if not os.path.exists(augmented_emotion_dir):
#         os.makedirs(augmented_emotion_dir)
    
#     # List all images in the emotion directory
#     image_files = os.listdir(emotion_dir)
    
#     # Collect all image paths for augmentation
#     image_paths = [os.path.join(emotion_dir, image_file) for image_file in image_files]
    
#     # Initialize a list to store valid image paths
#     valid_image_paths = []
    
#     # Count the number of valid images
#     for image_path in image_paths:
#         try:
#             io.imread(image_path)
#             valid_image_paths.append(image_path)
#         except:
#             print(f"Error reading {image_path}")
    
#     num_valid_images = len(valid_image_paths)
#     print(f"Number of valid images for {emotion}: {num_valid_images}")
    
#     # Calculate how many times each image should be augmented
#     num_augmentations_per_image = int((5020 - num_valid_images) / num_valid_images) + 1
    
#     # Augment images
#     for image_path in valid_image_paths:
#         try:
#             x = io.imread(image_path)
#             x = x.reshape((1, ) + x.shape)
#         except:
#             continue
        
#         for i in range(num_augmentations_per_image):
#             for batch in datagen.flow(x, batch_size=16, save_to_dir=augmented_emotion_dir, save_prefix='aug', save_format='png'):
#                 break  # Only one batch per image
        
#     print(f"Augmentation completed for {emotion}")

# print("All augmentations completed!")