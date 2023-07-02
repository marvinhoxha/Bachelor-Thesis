import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import pickle
import logging
import mlflow

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

LR = 6e-4
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = 244
NUM_CLASSES = 4

def get_train_generator():
    """Get the Train Path"""
    data_datagen = ImageDataGenerator(
        samplewise_center=True,
        rescale=1.0 / 255,
        width_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.15,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        fill_mode = 'reflect'
    )
    return data_datagen.flow_from_directory(
        "Effusion_detection_dataset/train/",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )


def get_valid_generator():
    """Get the Valid Path"""
    data_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return data_datagen.flow_from_directory(
        "Effusion_detection_dataset/valid/",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )


def create_xray_model(input_shape, num_classes):
    model = tf.keras.Sequential()

    # Convolutional layers
    model.add(tf.keras.layers.InputLayer(input_shape=(244, 244, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


def train():
    """Train the model"""
    logging.info("Training Model.")

    xray_model = create_xray_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    
    xray_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LR),
        loss=tf.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    train_generator = get_train_generator()
    valid_generator = get_valid_generator()

    logging.info(xray_model.summary())
    mlflow.autolog()

    xray_model.fit(
        train_generator, epochs=EPOCHS, validation_data=valid_generator
    )

    labels = train_generator.class_indices

    logging.info("Saving the model.")
    tf.saved_model.save(xray_model, "./models/Effusion_self/1")

    with open("./models/Effusion_labels.pickle", "wb") as handle:
        pickle.dump(labels, handle)
    logging.info("Finished training.")


if __name__ == "__main__":
    train()
