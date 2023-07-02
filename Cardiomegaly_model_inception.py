import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import regularizers
from PIL import ImageFile
import pickle
import os
import logging
import mlflow
from tensorflow.keras.applications import InceptionV3

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
LR = 6e-4
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = 244

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
        "Cardiomegaly_detection_dataset/train/",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )


def get_valid_generator():
    """Get the Valid Path"""
    data_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return data_datagen.flow_from_directory(
        "Cardiomegaly_detection_dataset/valid/",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )


def train():
    """Train the model"""
    logging.info("Training Model.")

    inception_body = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    inception_body.trainable = False
    unfreeze_layers = 20  # Number of blocks to unfreeze
    for layer in inception_body.layers[-unfreeze_layers:]:
        layer.trainable = True
    inception_model = tf.keras.Sequential([
        inception_body,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    inception_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LR),
        loss=tf.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    train_generator = get_train_generator()
    valid_generator = get_valid_generator()

    logging.info(inception_body.summary())
    logging.info("\n\n")
    logging.info(inception_model.summary())
    mlflow.autolog()

    inception_model.fit(
        train_generator, epochs=EPOCHS, validation_data=valid_generator
    )

    labels = train_generator.class_indices

    logging.info("Dump models.")
    inception_model.save("./models/Cardiomegaly_inception/1")

    with open("./models/Cardiomegaly_labels.pickle", "wb") as handle:
        pickle.dump(labels, handle)
    logging.info("Finished training.")


if __name__ == "__main__":
    train()
