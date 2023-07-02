import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import regularizers
from PIL import ImageFile
import pickle
import os
import logging
import mlflow
from tensorflow.keras.applications import DenseNet121

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
LR = 6e-4
BATCH_SIZE = 32
EPOCHS = 150
IMG_SIZE = 224

def get_train_generator():
    """Get the Train Path"""
    data_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
    )
    return data_datagen.flow_from_directory(
        "improved_dataset/train/",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )


def get_valid_generator():
    """Get the Valid Path"""
    data_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return data_datagen.flow_from_directory(
        "improved_dataset/valid/",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )


def train():
    """Train the model"""
    logging.info("Training Model.")

    densenet_body = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    densenet_body.trainable = False
    unfreeze_layers = 14  # Number of blocks to unfreeze
    for layer in densenet_body.layers[-unfreeze_layers:]:
        layer.trainable = True
    # inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # x = densenet_body(inputs, training=False)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
    # densenet_model = tf.keras.Model(inputs, outputs)
    densenet_model = tf.keras.Sequential([
        densenet_body,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    densenet_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LR),
        loss=tf.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    train_generator = get_train_generator()
    valid_generator = get_valid_generator()

    logging.info(densenet_body.summary())
    logging.info("\n\n")
    logging.info(densenet_model.summary())
    mlflow.autolog()

    densenet_model.fit(
        train_generator, epochs=EPOCHS, validation_data=valid_generator
    )

    labels = train_generator.class_indices

    logging.info("Dump models.")
    densenet_model.save("./models/XrayModel_densenet/1")

    with open("./models/labels.pickle", "wb") as handle:
        pickle.dump(labels, handle)
    logging.info("Finished training.")


if __name__ == "__main__":
    train()
