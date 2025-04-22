from google.colab import files
uploaded = files.upload()

from IPython import get_ipython
from IPython.display import display
from google.colab import files
import zipfile
import os

# Upload the zip file and get its name
uploaded = files.upload()
zip_path = list(uploaded.keys())[0]  # Get the name of the uploaded file

extract_folder = "extracted_files"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_folder)  # Extract to a folder

# List extracted files
os.listdir(extract_folder)

!pip install tensorflow numpy matplotlib seaborn scikit-learn

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = "extracted_files"  # This is the folder where your images are stored
batch_size = 32  # Define batch size
image_size = (64, 64)  # Resize images to a standard size

from tensorflow.keras.utils import image_dataset_from_directory

# Load dataset
dataset = image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="int",  # Integer labels for classification
    image_size=image_size,  # Resize images to a standard size
    batch_size=batch_size,
    shuffle=True  # Shuffle data for randomness
)

# Split into training and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Display class names (blood groups)
class_names = dataset.class_names
print("Classes detected:", class_names)

from collections import Counter

# Initialize counter for class distribution
class_counts = Counter()

# Iterate through dataset to count class occurrences
for images, labels in dataset.unbatch():
    class_counts[int(labels.numpy())] += 1  # Convert tensor to integer and count

# Print class distribution
print("Class Distribution:")
for i, count in class_counts.items():
    print(f"{class_names[i]}: {count}")

import os
dataset_path = "extracted_files/dataset_blood_group"
print("Folder contents:", os.listdir(dataset_path)[:10])  # Show first 10 items

from collections import Counter

class_names = dataset.class_names  # Get class names from the dataset
class_counts = Counter()

# Count the number of images per class
for _, labels in dataset.unbatch():
    class_counts[int(labels.numpy())] += 1

# Print the class distribution
print("Class Distribution:")
for i, count in sorted(class_counts.items()):
    print(f"{class_names[i]}: {count}")

dataset_path = "extracted_files/dataset_blood_group"

from tensorflow.keras.utils import image_dataset_from_directory

dataset = image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="int",
    image_size=(64, 64),  # Resize images
    batch_size=32,
    shuffle=True
)

import os

dataset_path = "extracted_files/dataset_blood_group"  # Update if needed

# Get all class folders inside the dataset path
class_names = sorted(os.listdir(dataset_path))
class_counts = {}

# Count images in each class folder
for class_name in class_names:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):  # Ensure it's a directory
        class_counts[class_name] = len(os.listdir(class_dir))

# Print class distribution
print("Class Distribution:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")


import matplotlib.pyplot as plt

def plot_class_distribution(class_names, class_counts):
    """
    Plots the distribution of classes in the dataset.

    Parameters:
    class_names (list): List of class names.
    class_counts (dict): Dictionary with class indices as keys and counts as values.

    Returns:
    None
    """

    # Extract class names and their corresponding counts
    classes = [class_names[i] for i in class_counts.keys()]
    counts = [class_counts[i] for i in class_counts.keys()]

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel("Blood Group")
    plt.ylabel("Count")
    plt.title("Class Distribution of Blood Groups")
    plt.xticks(rotation=45)
    plt.show()

# Example usage (replace with actual data)
class_names = ['A-', 'B+', 'AB-', 'O+', 'A+', 'AB+', 'B-', 'O-']
class_counts = {0: 1009, 1: 652, 2: 761, 3: 852, 4: 1200, 5: 800, 6: 741, 7: 712}

plot_class_distribution(class_names, class_counts)


import tensorflow as tf
import numpy as np
from collections import Counter
import random

def augment_image(image):
    """Apply random transformations to augment the image"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

def oversample_class(class_id, count, max_count, dataset):
    """Oversample a specific class using repetition and augmentation."""
    unbatched_dataset = dataset.unbatch()
    class_dataset = unbatched_dataset.filter(lambda img, lbl: tf.equal(lbl, class_id))

    # Convert dataset to list for augmentation if needed
    class_data = [(img.numpy(), lbl.numpy()) for img, lbl in class_dataset]

    if count < max_count:
        extra_needed = max_count - count
        augmented_images = []
        augmented_labels = []  # Separate list for labels

        for i in range(extra_needed):
            img, lbl = random.choice(class_data)
            img = augment_image(tf.convert_to_tensor(img, dtype=tf.float32))

            augmented_images.append(img)  # Append image to image list
            augmented_labels.append(lbl)  # Append label to label list

        # Convert back to TensorFlow datasets separately
        augmented_image_dataset = tf.data.Dataset.from_tensor_slices(augmented_images)
        augmented_label_dataset = tf.data.Dataset.from_tensor_slices(augmented_labels)

        # Zip the image and label datasets
        augmented_dataset = tf.data.Dataset.zip((augmented_image_dataset, augmented_label_dataset))

        class_dataset = class_dataset.concatenate(augmented_dataset)

    # Repeat the dataset to ensure we have enough samples
    repeat_factor = max_count // len(class_data) + int(max_count % len(class_data) > 0)
    class_dataset = class_dataset.repeat(repeat_factor)

    return class_dataset.take(max_count)


# Step 1: Balance the dataset
max_samples = max(class_counts.values())  # Maximum class count
balanced_datasets = [oversample_class(class_id, count, max_samples, dataset)
                     for class_id, count in class_counts.items()]

# Step 2: Merge datasets with a custom sampling function
def custom_sample(datasets):
    """Cycles through datasets, yielding one element from each."""
    iterators = [iter(d) for d in datasets]
    while True:
        try:
            for it in iterators:
                yield next(it)
        except StopIteration:
            break

balanced_dataset = tf.data.Dataset.from_generator(
    lambda: custom_sample(balanced_datasets),
    output_signature=(
        tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# Step 3: Verify final class distribution
balanced_class_counts = Counter([int(lbl.numpy()) for _, lbl in balanced_dataset])
print("Final Balanced Class Distribution:", balanced_class_counts)

import tensorflow as tf
import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt

# ... (previous code for oversampling remains the same) ...

# Step 3: Verify final class distribution and plot
balanced_class_counts = Counter([int(lbl.numpy()) for _, lbl in balanced_dataset])
print("Final Balanced Class Distribution:", balanced_class_counts)

# Plot the balanced class distribution
plt.figure(figsize=(10, 6))
plt.bar(balanced_class_counts.keys(), balanced_class_counts.values(), color='skyblue')
plt.xlabel("Blood Group")
plt.ylabel("Count")
plt.title("Balanced Class Distribution of Blood Groups")
plt.xticks(list(balanced_class_counts.keys()), class_names, rotation=45)  # Use class names for x-axis labels
plt.show()

# Instead of using unbatch directly, iterate through the dataset
# and manually extract and reshape the components

unbatched_data = []
for image, label in balanced_dataset:
    image = image.numpy()  # Convert image tensor to NumPy array
    label = label.numpy()  # Convert label tensor to NumPy array
    unbatched_data.append((image, label))

# Now unbatched_data is a list of (image, label) tuples

# Access the data:
for image, label in unbatched_data:
    print(image.shape, label.shape)  # Should output (64, 64, 3) and () for label

    dataset_size = 0
for _ in balanced_dataset:
    dataset_size += 1

print("Total dataset size:", dataset_size)

import tensorflow as tf

# ... (previous code for oversampling and balancing remains the same) ...

# Compute dataset size
dataset_size = 0
for _ in balanced_dataset:
    dataset_size += 1

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Compute sizes based on dataset size and desired splits
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = int(test_ratio * dataset_size)  # Calculate test_size

BATCH_SIZE = 32  # Define your batch size

# Split the dataset without unbatching
train_dataset = balanced_dataset.take(train_size)
val_test_dataset = balanced_dataset.skip(train_size)
val_dataset = val_test_dataset.take(val_size)
test_dataset = val_test_dataset.skip(val_size).take(test_size)  # Take test_size elements

# Rebatch the datasets after splitting
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

# Check the number of batches in each dataset
train_batch_count = sum(1 for _ in train_dataset)
val_batch_count = sum(1 for _ in val_dataset)
test_batch_count = sum(1 for _ in test_dataset)

print("Training dataset size:", train_batch_count * BATCH_SIZE)
print("Validation dataset size:", val_batch_count * BATCH_SIZE)
print("Testing dataset size:", test_batch_count * BATCH_SIZE)


import tensorflow as tf

def create_high_accuracy_model():
    """
    Creates a high-accuracy CNN model for blood group classification.

    Returns:
        model: The compiled Keras model.
    """

    inputs = tf.keras.Input(shape=(64, 64, 3))  # Define input layer explicitly

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Explicitly define the model

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create the model
high_acc_model = create_high_accuracy_model()

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define ReduceLROnPlateau callback to reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',     # Monitor validation loss
    factor=0.5,             # Reduce the learning rate by a factor of 0.5
    patience=3,             # Wait for 3 epochs without improvement before reducing LR
    verbose=1,              # Print a message when the learning rate is reduced
    min_lr=1e-6             # Minimum learning rate to avoid too small values
)

# Define EarlyStopping callback to stop training when validation loss doesn't improve
early_stop = EarlyStopping(
    monitor="val_loss",      # Monitor validation loss
    patience=5,              # Stop after 5 epochs without improvement
    verbose=1,               # Print a message when training is stopped
    restore_best_weights=True # Restore the model weights from the best epoch
)


history_high_acc = high_acc_model.fit(  # Fixed variable name (high_acc_modell → high_acc_model)
    train_dataset,
    validation_data=val_dataset,
    epochs=50,  # Adjust the number of epochs based on your preference
    callbacks=[reduce_lr, early_stop]  # Fixed missing '=' and square brackets syntax
)


# Evaluate the model on validation data
high_acc_eval = high_acc_model.evaluate(val_dataset)

# Print loss and accuracy
print(f"High Accuracy Model Loss: {high_acc_eval[0]}, Accuracy: {high_acc_eval[1]}")


import matplotlib.pyplot as plt

# Function to plot training & validation accuracy
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')  # Fixed typo from 'Hodel' to 'Model'
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Call the function to display the accuracy graph
plot_accuracy(history_high_acc)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Lists to store true and predicted labels
y_true = []  # True labels
y_pred = []  # Predicted labels

# Iterate over the test dataset and collect the true and predicted labels
for images, labels in test_dataset:
    predictions = high_acc_model.predict(images)  # Get model predictions
    predicted_labels = np.argmax(predictions, axis=1)  # Convert one-hot encoded predictions to class labels

    y_true.extend(labels.numpy())  # Convert tensor to numpy array and append
    y_pred.extend(predicted_labels)  # Append the predicted labels

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Save the model in HDF5 format
high_acc_model.save('model.h5')

print("Model saved in HDF5 format.")


pip show pillow
